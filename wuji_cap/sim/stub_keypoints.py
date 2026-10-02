"""STUB keypoints for developing B2 before A2 lands: synthetic MediaPipe-21 keypoints made from Wuji FK.

Not human data. Each hand follows a smooth joint trajectory with known SDK-order angles; the
keypoints are the Wuji FK points scaled up by ``1 / scale_true`` (a hand larger than the Wuji hand),
posed in the camera frame palm-down with fingers pointing away from the camera. The retargeter
should recover ``q_true`` and ``scale_true``; this tests frames, ordering, sides and gap handling,
not human-to-robot quality. T_cam_from_wrist is the Wuji wrist pose, so the stub's MANO-wrist ->
palm offset (AS-03) is the identity by construction.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from wuji_cap.common import frames
from wuji_cap.common import joint_order as jo
from wuji_cap.sim.hand_keypoints import WujiKinematics

N_FRAMES = 354  # session_032 length
FPS = 30.0
SCALE_TRUE = 0.9
# Frames marked invalid (keypoints NaN), to exercise gap handling.
INVALID = {"left": (100, 130), "right": (250, 260)}
# Wuji wrist axes in the camera frame for a palm-down pose (palm normal: right +y, left -y; fingers -z).
R_CAM_FROM_WRIST0 = {
    "right": np.array([[-1.0, 0, 0], [0, 1, 0], [0, 0, -1]]).T,
    "left": np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]]).T,
}
WRIST_POS0 = {"right": np.array([0.1, 0.03, 0.4]), "left": np.array([-0.1, 0.03, 0.4])}


def joint_trajectory(kin: WujiKinematics, t: np.ndarray, seed: int) -> np.ndarray:
    """[T, 20] SDK-order smooth sinusoids inside a comfortable part of each joint's range."""
    rng = np.random.default_rng(seed)
    lo, hi = kin.q_lower_sdk.copy(), kin.q_upper_sdk.copy()
    for f in jo.FINGERS[1:]:
        for j in (1, 3, 4):  # flexion joints: no hyperextension
            lo[jo.sdk_index(f, j)] = 0.0
        a = jo.sdk_index(f, 2)
        lo[a], hi[a] = -0.3, 0.3
    mid, amp = (lo + hi) / 2, 0.4 * (hi - lo)
    freq = rng.uniform(0.2, 0.5, jo.N_JOINTS)
    phase = rng.uniform(0, 2 * np.pi, jo.N_JOINTS)
    return mid + amp * np.sin(2 * np.pi * freq * t[:, None] + phase)


def wrist_trajectory(side: jo.Side, t: np.ndarray) -> np.ndarray:
    """[T, 4, 4] slow wrist motion around a palm-down pose in front of the camera."""
    rv = 0.3 * np.stack([np.sin(0.4 * t), np.sin(0.3 * t + 1), np.sin(0.25 * t + 2)], 1)
    R = Rotation.from_rotvec(rv).as_matrix() @ R_CAM_FROM_WRIST0[side]
    p = WRIST_POS0[side] + 0.05 * np.stack([np.sin(0.5 * t), np.sin(0.35 * t), np.sin(0.3 * t)], 1)
    return np.array([frames.make_T(R[i], p[i]) for i in range(len(t))])


def make_stub() -> tuple[dict, dict]:
    """Returns (arrays, params). Arrays follow contract v2 plus stub-only ground truth keys."""
    t = np.arange(N_FRAMES) / FPS
    arrays = {"frame_index": np.arange(N_FRAMES), "timestamps_s": t}
    for k, side in enumerate(jo.SIDES):
        kin = WujiKinematics(side)
        q = joint_trajectory(kin, t, seed=k)
        T_cw = wrist_trajectory(side, t)
        kp = np.array([frames.apply_T(T_cw[i], kin.keypoints21(q[i]) / SCALE_TRUE) for i in range(N_FRAMES)])
        valid = np.ones(N_FRAMES, bool)
        valid[slice(*INVALID[side])] = False
        kp[~valid] = np.nan
        T_cw[~valid] = np.nan
        # The stub wrist frame is the Wuji wrist frame at human scale: origin at the human wrist keypoint.
        arrays |= {
            f"{side}_kp21_cam": kp,
            f"{side}_T_cam_from_wrist": T_cw,
            f"{side}_valid": valid,
            f"{side}_betas": np.full(10, np.nan),  # no MANO in a stub
            f"stub_{side}_q_true_sdk": q,
        }
    params = {"stub": True, "n_frames": N_FRAMES, "fps": FPS, "scale_true": SCALE_TRUE,
              "invalid_frames": INVALID, "source": "Wuji Beta 2 URDF FK (not human data)"}
    return arrays, params
