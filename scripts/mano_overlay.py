"""A1 bilateral reconstruction audit: uv run python -m scripts.mano_overlay.

No A2 interface file is exported. Skeleton tips are labeled for visual review;
numerical reconstruction agreement is not a ground-truth fit error.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from wuji_cap.data.mano_clean import load_original, sha256
from wuji_cap.data.mano_reconstruct import FINGERS, finger_points, make_layer, numpy_reference, reconstruct
from wuji_cap.data.mano_session import load_session, project_camera


COLORS = {"left": (40, 225, 255), "right": (255, 170, 50)}  # OpenCV BGR


def label(image, text, xy, color=(255, 255, 255), scale=.6):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_hand(image, result, frame, camera, side):
    if not result.valid[frame]:
        return
    color = COLORS[side]
    for finger in FINGERS:
        pixels = project_camera(finger_points(result, frame, finger), camera)
        for a, b in zip(pixels[:-1], pixels[1:]):
            if np.isfinite([a, b]).all():
                cv2.line(image, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), color, 2, cv2.LINE_AA)
        for point in pixels:
            if np.isfinite(point).all():
                cv2.circle(image, tuple(np.rint(point).astype(int)), 3, color, -1, cv2.LINE_AA)
        if np.isfinite(pixels[-1]).all():
            xy = tuple(np.rint(pixels[-1] + [5, -5]).astype(int))
            label(image, side[0].upper() + ":" + finger, xy, color, .42)


def error_stats(delta, scale=1.):
    delta = np.asarray(delta) * scale
    return {"rms": float(np.sqrt(np.mean(delta ** 2))), "max_abs": float(np.max(np.abs(delta)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=Path("data/raw/ego-clinical-task/unscrew_cap/session_032"))
    parser.add_argument("--models", type=Path, default=Path("models/mano_v1_2_clean"))
    parser.add_argument("--originals", type=Path, default=Path("models/mano_v1_2/models"))
    parser.add_argument("--output", type=Path, default=Path("results/20260929-mano-a1"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    session = load_session(args.session)
    count = len(session.timestamps_s)
    camera = session.metadata["camera"]
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "schema_version": "mano_a1_audit_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "script": "scripts/mano_overlay.py", "session": str(args.session),
        "code_sha256": {str(path): sha256(path) for path in (
            Path("scripts/mano_overlay.py"), Path("wuji_cap/data/mano_session.py"),
            Path("wuji_cap/data/mano_reconstruct.py"), Path("wuji_cap/data/mano_clean.py"))},
        "source_sha256": session.source_sha256, "frame_count": count,
        "kinematic_only": True, "params": {"use_pca": False, "flat_hand_mean": True, "left_shapedirs_flip": False},
        "numerical_tolerance_m": 2e-7, "hands": {},
        "fit_accuracy": "No ground-truth landmarks in compact release; overlay review is qualitative.",
        "timestamps": {"duration_s": float(session.timestamps_s[-1]),
                       "min_step_s": float(np.diff(session.timestamps_s).min()) if count > 1 else None,
                       "max_step_s": float(np.diff(session.timestamps_s).max()) if count > 1 else None},
    }
    results = {}
    for side in ("left", "right"):
        hand = session.hands[side]
        layer = make_layer(args.models, side)
        results[side] = result = reconstruct(hand, layer)
        model = load_original(args.originals / f"MANO_{side.upper()}.pkl", session.metadata["mano"]["model_sha256"][side])
        valid_rows = np.flatnonzero(hand.valid)
        rows = valid_rows[np.linspace(0, len(valid_rows) - 1, min(12, len(valid_rows)), dtype=int)] if len(valid_rows) else []
        dv, dj, dp = [], [], []
        for row in rows:
            vertices, joints = numpy_reference(model, hand.betas, hand.hand_pose[row], hand.global_orient[row], hand.transl[row])
            dv.append(result.vertices[row] - vertices)
            dj.append(result.joints[row] - joints)
            dp.append(project_camera(result.joints[row], camera) - project_camera(joints, camera))
        checks = {
            "valid_frames": int(result.valid.sum()), "invalid_frames": int((~result.valid).sum()),
            "original_sha256": session.metadata["mano"]["model_sha256"][side],
            "clean_sha256": sha256(args.models / f"MANO_{side.upper()}.pkl"),
            "sample_frames": [int(row) for row in rows],
        }
        if len(rows):
            checks.update({"vertices_error_mm": error_stats(dv, 1000), "joints_error_mm": error_stats(dj, 1000),
                           "reference_reprojection_difference_px": error_stats(dp),
                           "positive_depth_all_vertices": bool((result.vertices[result.valid, :, 2] > 0).all())})
            assert max(np.max(np.abs(dv)), np.max(np.abs(dj))) < metrics["numerical_tolerance_m"]
        metrics["hands"][side] = checks
    left_flip = reconstruct(session.hands["left"], make_layer(args.models, "left", flip_left_shapedirs=True))
    left_valid = results["left"].valid
    if left_valid.any():
        metrics["left_shape_sign_diagnostic"] = {
            "flipped_minus_original_vertices_mm": error_stats(left_flip.vertices[left_valid] - results["left"].vertices[left_valid], 1000),
            "flipped_minus_original_projected_joints_px": error_stats(project_camera(left_flip.joints[left_valid], camera) - project_camera(results["left"].joints[left_valid], camera)),
            "production_convention": "Original model arrays; flip is diagnostic only. See visual_review.json when available.",
        }
    samples = set(np.linspace(0, count - 1, 6, dtype=int).tolist())
    capture = cv2.VideoCapture(str(args.session / "rgb.mp4"))
    video_path = args.output / f"{args.session.name}_overlay.mp4"
    writer = imageio.get_writer(str(video_path), fps=camera["fps_nominal"], codec="libx264", quality=8, macro_block_size=1)
    decoded = 0
    contact = []
    visibility = {side: {"frames_with_any_joint_in_image": 0, "frames_with_all_joints_in_image": 0} for side in results}
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if decoded >= count:
                raise ValueError("Video has more frames than MANO")
            if frame.shape[:2] != (camera["height"], camera["width"]):
                raise ValueError("Video dimensions differ from calibration")
            overlay = frame.copy()
            for side in ("left", "right"):
                draw_hand(overlay, results[side], decoded, camera, side)
                pixels = project_camera(results[side].joints[decoded], camera)
                inside = np.isfinite(pixels).all(axis=-1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < camera["width"]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < camera["height"])
                visibility[side]["frames_with_any_joint_in_image"] += int(inside.any())
                visibility[side]["frames_with_all_joints_in_image"] += int(inside.all())
            label(overlay, f"{args.session.name} | frame {decoded}/{count-1} | t={session.timestamps_s[decoded]:.3f}s | KINEMATIC ONLY", (25, 35))
            label(overlay, "LEFT yellow | RIGHT blue | original MANO shape convention", (25, 62))
            writer.append_data(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            if decoded in samples:
                contact.append(cv2.resize(overlay, (960, 540)))
                for side in ("left", "right"):
                    pixels = project_camera(results[side].vertices[decoded], camera)
                    if not np.isfinite(pixels).all():
                        continue
                    size = np.array([camera["width"], camera["height"]])
                    lo = np.clip(np.floor(pixels.min(axis=0) - 65).astype(int), 0, size)
                    hi = np.clip(np.ceil(pixels.max(axis=0) + 65).astype(int), 0, size)
                    if np.any(hi <= lo):
                        continue
                    choices = [("RGB", None), ("Original shapedirs", results[side])]
                    if side == "left":
                        choices.append(("Flipped x shapedirs", left_flip))
                    panels = []
                    for title, reconstruction in choices:
                        diagnostic = frame.copy()
                        if reconstruction is not None:
                            draw_hand(diagnostic, reconstruction, decoded, camera, side)
                        crop = diagnostic[lo[1]:hi[1], lo[0]:hi[0]]
                        # Preserve aspect ratio and reserve space for a readable heading.
                        factor = min(600 / crop.shape[1], 555 / crop.shape[0])
                        width, height = max(1, round(crop.shape[1] * factor)), max(1, round(crop.shape[0] * factor))
                        resized = cv2.resize(crop, (width, height))
                        panel = np.zeros((600, 600, 3), dtype=np.uint8)
                        x, y = (600 - width) // 2, 40 + (555 - height) // 2
                        panel[y:y+height, x:x+width] = resized
                        label(panel, f"Frame {decoded}: {title}", (12, 26), scale=.55)
                        panels.append(panel)
                    prefix = "left_sign" if side == "left" else "right_detail"
                    cv2.imwrite(str(args.output / f"{prefix}_{decoded:03d}.jpg"), np.hstack(panels))
            decoded += 1
    finally:
        capture.release()
        writer.close()
    if decoded != count:
        raise ValueError(f"Video has {decoded} frames, expected {count}")
    # Decode the produced video too; writer completion alone is not sufficient.
    verify = cv2.VideoCapture(str(video_path))
    encoded = 0
    while verify.read()[0]:
        encoded += 1
    verify.release()
    if encoded != count:
        raise ValueError(f"Overlay contains {encoded} frames, expected {count}")
    if len(contact) == 6:
        cv2.imwrite(str(args.output / "contact_sheet.jpg"), np.vstack([np.hstack(contact[i:i+2]) for i in (0, 2, 4)]))
    metrics["projected_visibility"] = visibility
    metrics["visibility_note"] = "Projection inside the image is not an occlusion test. valid means finite parameters, not a visible or accurate fit."
    metrics["video"] = {"input_decoded_frames": decoded, "output_decoded_frames": encoded,
                        "path": str(video_path), "sha256": sha256(video_path),
                        "fps": camera["fps_nominal"], "timing": "one output frame per source RGB frame at nominal fps; source timestamps shown"}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
