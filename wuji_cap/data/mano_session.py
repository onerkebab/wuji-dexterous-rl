"""Strict reader for compact_rgb_mano_v3; camera coordinates are meters.

Rows are never dropped or interpolated. An absent hand is expanded to NaNs;
a row with any nonfinite pose parameter is retained with valid=False.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .mano_clean import sha256


@dataclass
class HandParameters:
    hand_pose: np.ndarray
    global_orient: np.ndarray
    transl: np.ndarray
    betas: np.ndarray
    valid: np.ndarray


@dataclass
class ManoSession:
    path: Path
    metadata: dict
    timestamps_s: np.ndarray
    frame_index: np.ndarray
    hands: dict[str, HandParameters]
    source_sha256: dict[str, str]


def _array(archive, key, shape):
    value = archive[key]
    if value.shape != shape or value.dtype.kind != "f":
        raise ValueError(f"{key}: expected floating array {shape}, got {value.shape} {value.dtype}")
    return value.copy()


def load_session(path, verify_hashes=True):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata["format"] != "compact_rgb_mano_v3":
        raise ValueError("Expected compact_rgb_mano_v3")
    for key, expected in {
        "coordinate_frame": "zed_rectified_left_camera",
        "camera_axes": "+X right, +Y down, +Z forward",
        "translation_unit": "meter",
    }.items():
        if metadata[key] != expected:
            raise ValueError(f"Unsupported {key}: {metadata[key]}")
    if metadata["mano"]["use_pca"] is not False or metadata["mano"]["flat_hand_mean"] is not True:
        raise ValueError("Requires use_pca=False and flat_hand_mean=True")
    count = metadata["frame_count"]
    if type(count) is not int or count <= 0:
        raise ValueError("frame_count must be a positive integer")
    camera = metadata["camera"]
    for key in ("fx", "fy", "cx", "cy", "fps_nominal", "width", "height"):
        if not np.isfinite(camera[key]):
            raise ValueError(f"Nonfinite camera {key}")
    if min(camera[key] for key in ("fx", "fy", "fps_nominal", "width", "height")) <= 0:
        raise ValueError("Camera dimensions, focal lengths and fps must be positive")
    if np.any(np.asarray(camera["dist_coeffs"]) != 0):
        raise ValueError("Expected rectified camera with zero distortion")
    present = metadata["hands_present"]
    if len(set(present)) != len(present) or not set(present) <= {"left", "right"}:
        raise ValueError("Invalid hands_present")
    hashes = {}
    if verify_hashes:
        validation = json.loads((path / "validation.json").read_text())
        if validation["frame_count"] != count or validation["video_decoded_frames"] != count:
            raise ValueError("Validation frame count disagrees with metadata")
        for name in ("metadata.json", "mano.npz", "rgb.mp4"):
            hashes[name] = sha256(path / name)
            if hashes[name] != validation["output_sha256"][name]:
                raise ValueError(f"Session hash mismatch: {name}")
        hashes["validation.json"] = sha256(path / "validation.json")
    hands = {}
    with np.load(path / "mano.npz", allow_pickle=False) as archive:
        times = _array(archive, "timestamps_s", (count,))
        if not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0):
            raise ValueError("timestamps_s must be finite, start at zero, and strictly increase")
        for side in ("left", "right"):
            n = count if side in present else 0
            pose = _array(archive, side + "_hand_pose", (n, 45))
            orient = _array(archive, side + "_global_orient", (n, 3))
            transl = _array(archive, side + "_transl", (n, 3))
            betas = _array(archive, side + "_betas", (10,) if n else (0,))
            if not n:
                pose, orient, transl = (np.full((count, size), np.nan) for size in (45, 3, 3))
                betas = np.full(10, np.nan)
                valid = np.zeros(count, dtype=bool)
            else:
                if not np.isfinite(betas).all():
                    raise ValueError(f"{side}_betas must be finite for a present hand")
                valid = np.isfinite(np.concatenate((pose, orient, transl), axis=1)).all(axis=1)
                for value in (pose, orient, transl):
                    value[~valid] = np.nan
            hands[side] = HandParameters(pose, orient, transl, betas, valid)
    return ManoSession(path, metadata, times, np.arange(count), hands, hashes)


def project_camera(points, camera):
    """Pinhole projection; nonfinite and nonpositive-depth points stay NaN."""
    points = np.asarray(points)
    valid = np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    pixels = np.full(points.shape[:-1] + (2,), np.nan)
    pixels[valid] = points[valid, :2] / points[valid, 2, None]
    pixels[valid] *= [camera["fx"], camera["fy"]]
    pixels[valid] += [camera["cx"], camera["cy"]]
    return pixels
