"""Read/write derived npz files (interface contract v2, sprint doc "Interface contract v2")."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wuji_cap.common.joint_order import SIDES

SCHEMA_VERSION = 2
COMMON_KEYS = ("meta_json", "frame_index", "timestamps_s")
KEYPOINT_KEYS = tuple(f"{s}_{k}" for s in SIDES for k in ("kp21_cam", "T_cam_from_wrist", "valid", "betas"))


def write_npz(path: Path, arrays: dict, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"schema_version": SCHEMA_VERSION, **meta}
    np.savez_compressed(path, meta_json=np.array(json.dumps(meta, default=str)), **arrays)


def read_npz(path: Path) -> tuple[dict, dict]:
    """Returns (arrays, meta). Arrays exclude ``meta_json``."""
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta_json"}
        meta = json.loads(str(z["meta_json"]))
    return arrays, meta


def check_keypoints(arrays: dict) -> int:
    """Validate a keypoint file against contract v2. Returns T."""
    missing = [k for k in COMMON_KEYS[1:] + KEYPOINT_KEYS if k not in arrays]
    if missing:
        raise KeyError(f"keypoint file missing {missing}")
    T = len(arrays["timestamps_s"])
    shapes = {"kp21_cam": (T, 21, 3), "T_cam_from_wrist": (T, 4, 4), "valid": (T,), "betas": (10,)}
    for s in SIDES:
        for k, shp in shapes.items():
            if arrays[f"{s}_{k}"].shape != shp:
                raise ValueError(f"{s}_{k}: expected {shp}, got {arrays[f'{s}_{k}'].shape}")
        v = arrays[f"{s}_valid"]
        if v.dtype != bool:
            raise ValueError(f"{s}_valid must be bool")
        if not np.isfinite(arrays[f"{s}_kp21_cam"][v]).all():
            raise ValueError(f"{s}_kp21_cam has non-finite values on valid frames")
    if arrays["frame_index"].shape != (T,):
        raise ValueError("frame_index must be [T]")
    return T


def check_retarget(arrays: dict) -> int:
    """Validate a retarget file against contract v2 (extra keys allowed). Returns T."""
    T = len(arrays["timestamps_s"])
    shapes = {"valid": (T,), "q_wuji": (T, 20), "at_limit": (T, 20), "q_g1_arm": (T, 7),
              "ik_pos_err_m": (T,), "ik_rot_err_rad": (T,), "tip_err_m": (T, 5)}
    if arrays["wuji_joint_names"].shape != (20,) or arrays["frame_index"].shape != (T,):
        raise ValueError("wuji_joint_names must be [20] and frame_index [T]")
    for s in SIDES:
        if arrays[f"g1_arm_joint_names_{s}"].shape != (7,):
            raise ValueError(f"g1_arm_joint_names_{s} must be [7]")
        for k, shp in shapes.items():
            a = arrays[f"{s}_{k}"]
            if a.shape != shp:
                raise ValueError(f"{s}_{k}: expected {shp}, got {a.shape}")
        v = arrays[f"{s}_valid"]
        if v.dtype != bool or arrays[f"{s}_at_limit"].dtype != bool:
            raise ValueError(f"{s}_valid and {s}_at_limit must be bool")
        q = arrays[f"{s}_q_wuji"]
        if not np.isfinite(q[v]).all() or not np.isnan(q[~v]).all():
            raise ValueError(f"{s}_q_wuji must be finite exactly on valid frames")
        if arrays[f"{s}_at_limit"][~v].any():
            raise ValueError(f"{s}_at_limit set on an invalid frame")
        for k in ("q_g1_arm", "ik_pos_err_m", "ik_rot_err_rad"):  # NaN until B3; never set on invalid frames
            if np.isfinite(arrays[f"{s}_{k}"][~v]).any():
                raise ValueError(f"{s}_{k} set on an invalid frame")
        # B4 playback arrays (W1-B14, W1-B15), when present: finite everywhere, equal to the contract field on valid
        # frames except while catching up after a gap (the catch-up flag is set only on valid frames)
        for cmd, src, flag in ((f"{s}_q_wuji_cmd", f"{s}_q_wuji", f"{s}_wuji_catching_up"),
                               (f"{s}_q_g1_arm_cmd", f"{s}_q_g1_arm", f"{s}_arm_catching_up")):
            if cmd in arrays:
                c = arrays[cmd]
                if flag not in arrays:
                    raise ValueError(f"{cmd} needs {flag}")
                f = arrays[flag]
                if f.shape != v.shape or f.dtype != bool or f[~v].any():
                    raise ValueError(f"{flag} must be [T] bool and set only on valid frames")
                ok = v & ~f
                if c.shape != arrays[src].shape or not np.isfinite(c).all() or not np.array_equal(c[ok], arrays[src][ok]):
                    raise ValueError(f"{cmd} must be finite and equal {src} on valid frames that are not catching up")
        if f"{s}_held" in arrays and not np.array_equal(arrays[f"{s}_held"], ~v):
            raise ValueError(f"{s}_held must be the complement of {s}_valid")
    return T
