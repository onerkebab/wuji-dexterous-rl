"""Coordinate frames and rigid transforms. The only place the camera -> world convention is defined.

Camera frame (ZED rectified left, OpenCV optical): +X right, +Y down, +Z forward.
World frame (MuJoCo, used for every sim render): +X forward, +Y left, +Z up.
``R_WORLD_FROM_CAM`` maps optical axes onto world axes: cam +Z -> world +X, cam +X -> world -Y,
cam +Y -> world -Z. Where the camera sits on the G1 is ``T_torso_from_cam`` (B3, AS-01).

Quaternions: MuJoCo uses (w, x, y, z); SciPy uses (x, y, z, w). Convert only with the helpers here.
"""

from __future__ import annotations

import numpy as np

R_WORLD_FROM_CAM = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)


# A ROS-style sensor link (x forward, y left, z up) relates to its optical frame exactly as the world
# frame above relates to the camera.
R_LINK_FROM_OPTICAL = R_WORLD_FROM_CAM

# AS-01: the ZED rectified-left camera stands in for the G1 head camera, mounted like the D435.
# Unitree g1_29dof_rev_1_0.urdf (TWIST2 @ 603cba1) `d435_joint`: torso_link -> d435_link.
D435_XYZ_IN_TORSO = np.array([0.0576235, 0.01753, 0.42987])
D435_RPY_IN_TORSO = np.array([0.0, 0.8307767239493009, 0.0])  # URDF fixed-axis roll, pitch, yaw


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def T_world_from_cam(cam_pos_world: np.ndarray = np.zeros(3)) -> np.ndarray:
    """Camera at ``cam_pos_world`` looking along world +X, image up = world +Z."""
    return make_T(R_WORLD_FROM_CAM, np.asarray(cam_pos_world, dtype=float))


def rot_from_urdf_rpy(rpy: np.ndarray) -> np.ndarray:
    """URDF ``rpy``: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def T_torso_from_cam() -> np.ndarray:
    """Optical camera frame (OpenCV, as in the keypoint files) -> G1 torso_link frame (AS-01)."""
    T_torso_from_link = make_T(rot_from_urdf_rpy(D435_RPY_IN_TORSO), D435_XYZ_IN_TORSO)
    return T_torso_from_link @ make_T(R_LINK_FROM_OPTICAL, np.zeros(3))


def apply_T(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to points ``p[..., 3]``."""
    p = np.asarray(p, dtype=float)
    return p @ T[:3, :3].T + T[:3, 3]


def inv_T(T: np.ndarray) -> np.ndarray:
    """Inverse of 4x4 rigid transforms ``T[..., 4, 4]``."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    out = np.zeros_like(T)
    Rt = np.swapaxes(R, -1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


def quat_wxyz_from_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    return np.concatenate([q[..., 3:], q[..., :3]], axis=-1)


def quat_xyzw_from_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    return np.concatenate([q[..., 1:], q[..., :1]], axis=-1)


def rotation_angle(R: np.ndarray) -> np.ndarray:
    """Rotation angle (rad) of rotation matrices ``R[..., 3, 3]``."""
    c = (np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(c, -1.0, 1.0))


def umeyama(src: np.ndarray, dst: np.ndarray, scale: float | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity ``dst ~ s * R @ src + t`` (Umeyama 1991), proper rotation (det +1).

    ``src``, ``dst``: [N, 3] corresponding points, N >= 3, not collinear. Returns (s, R, t).
    ``scale=None`` estimates s; a number fixes it (1.0 gives a rigid fit). R does not depend on s.
    Never returns a reflection: a mirrored point set (e.g. a left hand fitted to a right-hand
    template) gives a large residual instead.
    """
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or src.shape[0] < 3:
        raise ValueError(f"expected two [N>=3, 3] arrays, got {src.shape} and {dst.shape}")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (xs**2).sum(1).mean()) if scale is None else float(scale)
    t = mu_d - s * R @ mu_s
    return s, R, t
