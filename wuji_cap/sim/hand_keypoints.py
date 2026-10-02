"""MediaPipe-21 keypoints on the Wuji Beta 2 hand, and the per-frame palm fit (human -> Wuji wrist frame).

Each MediaPipe keypoint is matched to a Beta 2 URDF frame at the corresponding joint origin:
wrist -> ``wrist``; thumb CMC/MCP/IP/tip -> ``thumb_{proximal,middle,distal,tip}``;
finger MCP/PIP/DIP/tip -> ``<finger>_{proximal,middle,distal,tip}``. ``proximal`` is the child of
the flexion joint (MCP flex; thumb CMC flex), ``middle`` of the PIP (thumb MCP), ``distal`` of the
DIP (thumb IP). The Beta 2 MCP abduction axis sits 14 mm distal to the flexion axis; the human
MCP is one point, matched to the flexion origin.

Palm fit (W1-18, extended per frame in W1-B05): a similarity fit of the human palm keypoints
``PALM_KP`` (wrist + 5 knuckles) onto the same points of the Wuji zero pose. These points do not
move with any Wuji joint, so the template is pose-independent.

Direction targets (W1-B08): the human fingers are not proportional to the Wuji fingers (after the
palm scale, human MCP->PIP is ~0.4-0.55x the Wuji length), so the retargeter gets each human segment's
direction with the Wuji segment length, chained from the Wuji knuckle (thumb: CMC). The first
segment's Wuji length depends on abduction, so lengths come from a reference pose (previous frame).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from wuji_cap.common import frames
from wuji_cap.common import joint_order as jo
from wuji_cap.common.joint_order import Side
from wuji_cap.sim import wuji_model as wm

# MediaPipe hand landmark order.
MP_NAMES = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
MP_TIPS = (4, 8, 12, 16, 20)
PALM_KP = (0, 1, 5, 9, 13, 17)  # wrist + thumb CMC + 4 finger MCPs

_URDF_FINGER = {
    "thumb": "thumb",
    "index": "index_finger",
    "middle": "middle_finger",
    "ring": "ring_finger",
    "pinky": "pinky",
}


def mp_frame_names(side: Side) -> tuple[str, ...]:
    """Beta 2 URDF frame name for each of the 21 MediaPipe keypoints."""
    p = jo.side_prefix(side)
    out = [p + "wrist"]
    for f in jo.FINGERS:
        out += [f"{p}{_URDF_FINGER[f]}_{s}" for s in ("proximal", "middle", "distal", "tip")]
    return tuple(out)


class WujiKinematics:
    """Pinocchio FK of the Beta 2 URDF, with SDK-order joint input."""

    def __init__(self, side: Side):
        self.side = side
        self.model = pin.buildModelFromUrdf(str(wm.urdf_path(side)))
        self.data = self.model.createData()
        self.pin_joint_names = tuple(self.model.names[1:])
        if self.model.nq != jo.N_JOINTS or len(self.pin_joint_names) != jo.N_JOINTS:
            raise ValueError(f"expected 20 one-DoF joints, got nq={self.model.nq}")
        self.frame_ids = [self._frame_id(n) for n in mp_frame_names(side)]
        self.q_lower_sdk = jo.to_sdk(self.model.lowerPositionLimit, self.pin_joint_names, side)
        self.q_upper_sdk = jo.to_sdk(self.model.upperPositionLimit, self.pin_joint_names, side)
        self.zero_kp = self.keypoints21(np.zeros(jo.N_JOINTS))
        self.palm_template = self.zero_kp[list(PALM_KP)]

    def _frame_id(self, name: str) -> int:
        i = self.model.getFrameId(name)
        if i >= self.model.nframes:
            raise KeyError(f"frame {name!r} not in {wm.urdf_path(self.side)}")
        return i

    def keypoints21(self, q_sdk: np.ndarray) -> np.ndarray:
        """[21, 3] MediaPipe-order keypoints (m) in the Wuji wrist frame."""
        q_pin = jo.from_sdk(np.asarray(q_sdk, dtype=float), self.pin_joint_names, self.side)
        pin.framesForwardKinematics(self.model, self.data, q_pin)
        return np.array([self.data.oMf[i].translation for i in self.frame_ids])


@dataclass
class PalmFit:
    T_cam_from_palm: np.ndarray  # [4, 4] rigid; "palm" = Wuji wrist frame
    scale: float  # Wuji size / human size used for this fit
    rms_m: float  # RMS palm-point residual after fit, in Wuji (scaled) units
    kp_palm: np.ndarray  # [21, 3] human keypoints mapped into the Wuji wrist frame, scaled by `scale`


def palm_scale(kp21_cam: np.ndarray, template: np.ndarray) -> float:
    """Wuji/human size ratio from a similarity fit of the palm points of one frame."""
    s, _, _ = frames.umeyama(np.asarray(kp21_cam)[list(PALM_KP)], template)
    return s


def fit_palm(kp21_cam: np.ndarray, template: np.ndarray, scale: float) -> PalmFit:
    """Map one frame of human keypoints (camera frame) into the Wuji wrist frame at a fixed scale."""
    kp = np.asarray(kp21_cam, dtype=float)
    s, R, t = frames.umeyama(kp[list(PALM_KP)], template, scale=scale)
    kp_palm = s * kp @ R.T + t
    rms = float(np.sqrt(((kp_palm[list(PALM_KP)] - template) ** 2).sum(1).mean()))
    # Wuji-frame point x maps to camera point R^T (x - t) / s; the rigid part is the palm pose.
    T_cam_from_palm = frames.make_T(R.T, -R.T @ t / s)
    return PalmFit(T_cam_from_palm=T_cam_from_palm, scale=s, rms_m=rms, kp_palm=kp_palm)


def finger_chains() -> list[list[int]]:
    """MediaPipe indices per finger, knuckle (thumb: CMC) to tip."""
    return [[1 + 4 * f + j for j in range(4)] for f in range(5)]


def direction_targets(kp_palm: np.ndarray, wuji_ref_kp: np.ndarray) -> np.ndarray:
    """[21, 3] targets in the Wuji wrist frame: wrist and knuckles at their Wuji positions (``PALM_KP``,
    pose-independent), then each human segment's direction with the Wuji segment length taken from
    ``wuji_ref_kp`` (Wuji keypoints at a reference pose). Lengths are fixed except the first segment
    of each chain: the MCP abduction axis (thumb: CMC abduction) is offset from the flexion origin,
    so knuckle -> next shortens with abduction (index 52.1 -> 50.8 mm, thumb 44.4 -> 39.8 mm at
    0.5 rad). The retargeter passes the previous frame's solution as the reference."""
    x = np.asarray(kp_palm, dtype=float)
    ref = np.asarray(wuji_ref_kp, dtype=float)
    y = ref.copy()
    for c in finger_chains():
        for a, b in zip(c[:-1], c[1:]):
            d = x[b] - x[a]
            y[b] = y[a] + d / np.linalg.norm(d) * np.linalg.norm(ref[b] - ref[a])
    return y


def bend_angles(kp21: np.ndarray) -> np.ndarray:
    """[..., 5, 3] bend angles (rad) per finger at the knuckle, middle and distal joints of the chain
    wrist -> knuckle -> ... -> tip (fingers: MCP, PIP, DIP; thumb: CMC, MCP, IP). Angle between
    consecutive segments; 0 = straight. Unsigned, so a hyperextension also reads as a bend."""
    kp = np.asarray(kp21, dtype=float)
    out = []
    for c in finger_chains():
        pts = [0] + c
        u = kp[..., pts[1:4], :] - kp[..., pts[0:3], :]
        v = kp[..., pts[2:5], :] - kp[..., pts[1:4], :]
        cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
        out.append(np.arccos(np.clip(cos, -1.0, 1.0)))
    return np.stack(out, axis=-2)
