"""Contract-v2 MANO keypoints. Run with uv run python -B -m wuji_cap.data.mano_keypoints.

The wrist frame is rooted at the shaped MANO wrist, with axes rotated by
global_orient. Thus p_cam = R_cam_from_wrist @ p_wrist + reconstructed_wrist.
It is not the Wuji palm frame. valid describes reconstruction, not image fit.
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from .mano_clean import sha256
from .mano_reconstruct import FINGERS, make_layer, reconstruct
from .mano_session import load_session


SIDES = ("left", "right")
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
VALIDITY_NOTE = "Finite reconstructable parameters and geometry; not visibility, occlusion or anatomical fit quality."
FIT_NOTE = (
    "AS-A02 remains OPEN under AS-06/AS-07: numerical reconstruction is verified, "
    "but anatomical image accuracy is not. Retain original left shapedirs; visible "
    "ring/pinky discrepancies are unresolved. AS-05 fingertip accuracy also remains OPEN."
)


def keypoints_from_reconstruction(hand, result):
    """Return kp21, wrist transforms and valid, masking entire invalid rows."""
    points = np.concatenate(
        [result.joints[:, :1]] + [
            np.concatenate((result.joints[:, list(FINGERS[name][0][1:])],
                            result.vertices[:, FINGERS[name][1], None]), axis=1)
            for name in FINGER_ORDER
        ], axis=1,
    )
    valid = (hand.valid & result.valid & np.isfinite(points).all(axis=(1, 2))
             & np.isfinite(hand.global_orient).all(axis=1))
    transforms = np.full((len(valid), 4, 4), np.nan, dtype=np.float64)
    for row in np.flatnonzero(valid):
        transforms[row] = np.eye(4)
        transforms[row, :3, :3] = cv2.Rodrigues(hand.global_orient[row].astype(np.float64))[0]
        transforms[row, :3, 3] = points[row, 0]
    points[~valid] = np.nan
    return points, transforms, valid


def validate_keypoints(arrays):
    """Validate the public contract without pickle or model access."""
    expected = {"meta_json", "frame_index", "timestamps_s"} | {
        f"{side}_{name}" for side in SIDES
        for name in ("kp21_cam", "T_cam_from_wrist", "valid", "betas")
    }
    if set(arrays) != expected:
        raise ValueError("Contract v2 requires exactly the documented keys")
    raw_meta = np.asarray(arrays["meta_json"])
    if raw_meta.shape != () or raw_meta.dtype.kind != "U":
        raise ValueError("meta_json must be a scalar Unicode string")
    meta = json.loads(raw_meta.item())
    if meta.get("schema_version") != 2:
        raise ValueError("Expected schema_version 2")
    required = {"sources", "script", "git_commit", "git_dirty", "created_utc", "params"}
    if not required <= meta.keys():
        raise ValueError("Missing provenance fields")
    frames, times = arrays["frame_index"], arrays["timestamps_s"]
    if frames.ndim != 1 or frames.dtype.kind not in "iu" or not len(frames):
        raise ValueError("frame_index must be a nonempty integer vector")
    count = len(frames)
    if frames[0] < 0 or np.any(frames[1:] <= frames[:-1]):
        raise ValueError("frame_index must be nonnegative and strictly increasing")
    if (times.shape != (count,) or times.dtype.kind != "f" or not np.isfinite(times).all()
            or times[0] < 0 or np.any(np.diff(times) <= 0)):
        raise ValueError("Invalid timestamps_s")
    for side in SIDES:
        valid = arrays[f"{side}_valid"]
        if valid.shape != (count,) or valid.dtype != np.dtype(bool):
            raise ValueError(f"{side}_valid must be [T] bool")
        for name, shape in (("kp21_cam", (count, 21, 3)), ("T_cam_from_wrist", (count, 4, 4)), ("betas", (10,))):
            value = arrays[f"{side}_{name}"]
            if value.shape != shape or value.dtype.kind != "f":
                raise ValueError(f"Invalid {side}_{name} shape or dtype")
            if name == "betas":
                if not (np.isfinite(value).all() or (not valid.any() and np.isnan(value).all())):
                    raise ValueError(f"Invalid {side}_betas")
            elif not (np.isfinite(value[valid]).all() and np.isnan(value[~valid]).all()):
                raise ValueError(f"{side}_{name} must be finite on valid rows, NaN otherwise")
        transform = arrays[f"{side}_T_cam_from_wrist"][valid]
        rotation = transform[:, :3, :3]
        if not (np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-6, rtol=0)
                and np.allclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)
                and np.allclose(transform[:, 3], [0, 0, 0, 1], atol=1e-7, rtol=0)
                and np.allclose(transform[:, :3, 3], arrays[f"{side}_kp21_cam"][valid, 0], atol=1e-7, rtol=0)):
            raise ValueError(f"Invalid {side} wrist transform")
    return meta


def export_session(session_path, destination, models=Path("models/mano_v1_2_clean"),
                   originals=Path("models/mano_v1_2/models"), batch_size=64):
    """Verify sources, reconstruct, validate and atomically publish a v2 NPZ."""
    session = load_session(session_path)
    models, originals, destination = Path(models), Path(originals), Path(destination)
    sources = [{"path": str(session.path / name), "sha256": digest}
               for name, digest in session.source_sha256.items()]
    for side in SIDES:
        name = f"MANO_{side.upper()}.pkl"
        digest = sha256(originals / name)
        if digest != session.metadata["mano"]["model_sha256"][side]:
            raise ValueError(f"Original MANO hash mismatch: {side}")
        sources.extend([{"path": str(originals / name), "sha256": digest},
                        {"path": str(models / name), "sha256": sha256(models / name)}])
    repo = Path(__file__).resolve().parents[2]
    code_paths = [Path(__file__), Path(__file__).with_name("mano_reconstruct.py"),
                  Path(__file__).with_name("mano_session.py"), Path(__file__).with_name("mano_clean.py")]
    meta = {
        "schema_version": 2, "artifact_type": "mano_keypoints", "sources": sources,
        "script": "wuji_cap/data/mano_keypoints.py",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": {str(p.resolve().relative_to(repo)): sha256(p) for p in code_paths},
        "versions": {name: version(name) for name in ("numpy", "torch", "smplx", "opencv-python-headless")},
        "params": {"use_pca": False, "flat_hand_mean": True, "left_shapedirs_flip": False,
                   "batch_size": batch_size, "interpolate": False,
                   "finger_order": list(FINGER_ORDER),
                   "native_joint_chains": {name: list(FINGERS[name][0][1:]) for name in FINGER_ORDER},
                   "tip_vertices": {name: FINGERS[name][1] for name in FINGER_ORDER}},
        "coordinate_frame": session.metadata["coordinate_frame"], "camera_axes": session.metadata["camera_axes"],
        "units": "meter", "keypoint_order": "MediaPipe-21",
        "frame_index_note": "Zero-based frame in this session's rgb.mp4, not the original recording's frame number.",
        "wrist_frame": "Origin: reconstructed MANO joint 0 (shaped rest wrist + transl). Axes: global_orient rotation. p_cam = R @ p_wrist + wrist_cam.",
        "validity": VALIDITY_NOTE, "kinematic_only": True,
        "open_assumptions": ["AS-A02", "AS-05", "AS-06", "AS-07"], "fit_caveat": FIT_NOTE,
        "review_evidence": ["results/20260929-mano-a1/visual_review.json", "results/20260930-mano-session021/visual_review.json"],
    }
    arrays = {"meta_json": np.array(json.dumps(meta, allow_nan=False, sort_keys=True)),
              "frame_index": session.frame_index.copy(), "timestamps_s": session.timestamps_s.copy()}
    for side in SIDES:
        hand = session.hands[side]
        result = reconstruct(hand, make_layer(models, side), batch_size=batch_size)
        points, transform, valid = keypoints_from_reconstruction(hand, result)
        arrays.update({f"{side}_kp21_cam": points, f"{side}_T_cam_from_wrist": transform,
                       f"{side}_valid": valid, f"{side}_betas": hand.betas.copy()})
    validate_keypoints(arrays)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Avoid exposing a partially written archive to B's reader.
    import tempfile
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        with np.load(temporary, allow_pickle=False) as stored:
            validate_keypoints(stored)
            for name in arrays:
                np.testing.assert_array_equal(stored[name], arrays[name])
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=Path("data/raw/ego-clinical-task/unscrew_cap/session_032"))
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--models", type=Path, default=Path("models/mano_v1_2_clean"))
    parser.add_argument("--originals", type=Path, default=Path("models/mano_v1_2/models"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(2)
    destination = args.destination or Path("data/derived/keypoints") / f"{args.session.name}.npz"
    print(export_session(args.session, destination, args.models, args.originals, args.batch_size))


if __name__ == "__main__":
    main()
