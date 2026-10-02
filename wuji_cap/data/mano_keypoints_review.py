"""Audit the serialized A2 NPZ and render labeled review plots from its keypoints.

uv run python -B -m wuji_cap.data.mano_keypoints_review
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np

from .mano_clean import load_original, sha256
from .mano_keypoints import FIT_NOTE, SIDES, VALIDITY_NOTE, validate_keypoints
from .mano_reconstruct import numpy_reference
from .mano_session import load_session, project_camera


COLORS = {"left": (40, 225, 255), "right": (255, 170, 50)}
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def label(image, text, xy, color=(255, 255, 255), scale=.55):
    for width, ink in ((4, (0, 0, 0)), (1, color)):
        cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, ink, width, cv2.LINE_AA)


def draw(image, pixels, side):
    color = COLORS[side]
    for finger, name in enumerate(FINGERS):
        chain = [0, *range(1 + 4 * finger, 5 + 4 * finger)]
        for a, b in zip(chain[:-1], chain[1:]):
            if np.isfinite(pixels[[a, b]]).all():
                cv2.line(image, tuple(np.rint(pixels[a]).astype(int)), tuple(np.rint(pixels[b]).astype(int)), color, 2, cv2.LINE_AA)
        tip = chain[-1]
        if np.isfinite(pixels[tip]).all():
            label(image, f"{side[0].upper()} {tip}:{name}", tuple(np.rint(pixels[tip] + [5, -8]).astype(int)), color, .45)
    for i, point in enumerate(pixels):
        if np.isfinite(point).all():
            xy = tuple(np.rint(point).astype(int))
            cv2.circle(image, xy, 3, color, -1, cv2.LINE_AA)
            if i not in (4, 8, 12, 16, 20):
                label(image, str(i), (xy[0] + 4, xy[1] + 12), color, .35)


def audit(session_path, keypoints_path, output, originals):
    session = load_session(session_path)
    with np.load(keypoints_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    meta = validate_keypoints(arrays)
    np.testing.assert_array_equal(arrays["frame_index"], session.frame_index)
    np.testing.assert_array_equal(arrays["timestamps_s"], session.timestamps_s)
    for name, digest in session.source_sha256.items():
        assert {"path": str(session.path / name), "sha256": digest} in meta["sources"]
    output.mkdir(parents=True, exist_ok=True)
    count = len(session.frame_index)
    camera = session.metadata["camera"]
    metrics = {
        "schema_version": "mano_a2_audit_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "wuji_cap/data/mano_keypoints_review.py", "code_sha256": sha256(Path(__file__)),
        "keypoints": {"path": str(keypoints_path), "sha256": sha256(keypoints_path)},
        "export_metadata": meta, "frame_count": count, "frame_index_preserved": True,
        "timestamps_preserved": True, "kinematic_only": True, "fit_caveat": FIT_NOTE,
        "validity": VALIDITY_NOTE, "numerical_tolerance_m": 2e-7, "hands": {},
    }
    for side in SIDES:
        hand = session.hands[side]
        points, transform, valid = (arrays[f"{side}_{name}"] for name in ("kp21_cam", "T_cam_from_wrist", "valid"))
        np.testing.assert_array_equal(arrays[f"{side}_betas"], hand.betas)
        model = load_original(originals / f"MANO_{side.upper()}.pkl", session.metadata["mano"]["model_sha256"][side])
        available = np.flatnonzero(valid)
        rows = available[np.linspace(0, len(available) - 1, min(12, len(available)), dtype=int)] if len(available) else []
        errors, wrist_errors = [], []
        for row in rows:
            vertices, joints = numpy_reference(model, hand.betas, hand.hand_pose[row], hand.global_orient[row], hand.transl[row])
            # Independent explicit MP21 layout, including the five mesh tips.
            expected = np.vstack([joints[0], joints[13:16], vertices[744], joints[1:4], vertices[320],
                                  joints[4:7], vertices[443], joints[10:13], vertices[554], joints[7:10], vertices[671]])
            errors.append(float(np.max(np.abs(points[row] - expected))))
            wrist_errors.append(float(np.max(np.abs(transform[row, :3, 3] - joints[0]))))
            np.testing.assert_allclose(points[row], expected, rtol=0, atol=2e-7)
            np.testing.assert_allclose(transform[row, :3, :3], cv2.Rodrigues(hand.global_orient[row].astype(float))[0], rtol=0, atol=1e-12)
        pixels = project_camera(points, camera)
        inside = np.isfinite(pixels).all(axis=-1) & (pixels[..., 0] >= 0) & (pixels[..., 0] < camera["width"]) & (pixels[..., 1] >= 0) & (pixels[..., 1] < camera["height"])
        metrics["hands"][side] = {
            "valid_frames": int(valid.sum()), "invalid_frames": int((~valid).sum()),
            "source_valid_frames": int(hand.valid.sum()), "reconstruction_rejected_frames": int((hand.valid & ~valid).sum()),
            "sample_frames": [int(row) for row in rows],
            "kp21_reference_max_abs_error_m": max(errors) if errors else None,
            "wrist_reference_max_abs_error_m": max(wrist_errors) if wrist_errors else None,
            "all_21_project_inside_image_frames": int(inside.all(axis=1).sum()),
            "projection_note": "Image bounds only, not an occlusion or fit accuracy check.",
        }
    samples = sorted(set(np.linspace(0, count - 1, min(count, 6), dtype=int).tolist()))
    capture = cv2.VideoCapture(str(session.path / "rgb.mp4"))
    plots, contact = [], []
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if decoded >= count or frame.shape[:2] != (camera["height"], camera["width"]):
                raise ValueError("RGB frame count or dimensions mismatch")
            if decoded in samples:
                overlay = frame.copy()
                for side in SIDES:
                    pixels = project_camera(arrays[f"{side}_kp21_cam"][decoded], camera)
                    draw(overlay, pixels, side)
                    finite = pixels[np.isfinite(pixels).all(axis=1)]
                    if not len(finite):
                        continue
                    size = np.array([camera["width"], camera["height"]])
                    lo = np.clip(np.floor(finite.min(axis=0) - 100).astype(int), 0, size)
                    hi = np.clip(np.ceil(finite.max(axis=0) + 100).astype(int), 0, size)
                    if np.any(hi <= lo):
                        continue
                    decorated = frame.copy()
                    draw(decorated, pixels, side)
                    panels = []
                    for title, source in (("RGB", frame), ("Exported MP21", decorated)):
                        crop = source[lo[1]:hi[1], lo[0]:hi[0]]
                        factor = min(700 / crop.shape[1], 640 / crop.shape[0])
                        resized = cv2.resize(crop, (max(1, round(crop.shape[1] * factor)), max(1, round(crop.shape[0] * factor))))
                        panel = np.zeros((700, 700, 3), np.uint8)
                        panel[50:50 + resized.shape[0], :resized.shape[1]] = resized
                        label(panel, f"{side} frame {decoded} | {title}", (12, 30))
                        panels.append(panel)
                    filename = f"{side}_{decoded:03d}.jpg"
                    if not cv2.imwrite(str(output / filename), np.hstack(panels)):
                        raise OSError(f"Could not write {filename}")
                    plots.append(filename)
                label(overlay, f"frame {decoded} | {session.timestamps_s[decoded]:.3f}s | KINEMATIC ONLY | AS-A02 OPEN", (25, 35))
                label(overlay, "Exported MediaPipe-21 | left yellow, right blue | valid is not image-fit quality", (25, 62))
                contact.append(cv2.resize(overlay, (960, 540)))
            decoded += 1
    finally:
        capture.release()
    if decoded != count:
        raise ValueError(f"Decoded {decoded} RGB frames, expected {count}")
    if len(contact) % 2:
        contact.append(np.zeros_like(contact[0]))
    if not cv2.imwrite(str(output / "contact_sheet.jpg"), np.vstack([np.hstack(contact[i:i+2]) for i in range(0, len(contact), 2)])):
        raise OSError("Could not write contact sheet")
    plots.append("contact_sheet.jpg")
    metrics.update({"rgb_decoded_frames": decoded, "plot_frames": samples,
                    "plots": {name: sha256(output / name) for name in plots}})
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"metrics": str(output / "metrics.json"), "hands": metrics["hands"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=Path("data/raw/ego-clinical-task/unscrew_cap/session_032"))
    parser.add_argument("--keypoints", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/20260930-mano-a2"))
    parser.add_argument("--originals", type=Path, default=Path("models/mano_v1_2/models"))
    args = parser.parse_args()
    keypoints = args.keypoints or Path("data/derived/keypoints") / f"{args.session.name}.npz"
    audit(args.session, keypoints, args.output, args.originals)


if __name__ == "__main__":
    main()
