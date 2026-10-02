"""Canonical Wuji Hand 2 joint order and conversions to every other ordering.

Canonical order = Wuji SDK flat joint index 0..19 (DECISIONS W1-05, spec section 3 principle 1):
finger-major (thumb, index, middle, ring, pinky), joint1..joint4 within each finger, proximal
to distal. The SDK labels these ``thumb_joint1`` .. ``pinky_joint4`` (``WujiHandJointHandle.label``).

SDK joint j of each finger is the j-th joint of that finger's kinematic chain in the Beta 2
model: flex, abd, pip/mcp, dip/ip (ASSUMPTIONS AS-B01). The Beta 2 MJCF and URDF share joint
names, e.g. ``r_index_finger_mcp_abd``; the MJCF actuators are ``r_FFJ1`` etc.

Other orderings (MJCF qpos, Pinocchio/URDF as used by wuji-retargeting, which is alphabetical
by finger, TWIST2, device) are given as lists of those model joint names, and the permutation
is always built from names, never from hand-typed indices.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

Side = Literal["left", "right"]
SIDES: tuple[Side, ...] = ("left", "right")

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
N_JOINTS = 20

SDK_JOINT_NAMES: tuple[str, ...] = tuple(f"{f}_joint{j}" for f in FINGERS for j in range(1, 5))

# Beta 2 model joint names (without side prefix), per finger, proximal to distal.
_MODEL_JOINTS = {
    "thumb": ("thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip"),
    "index": tuple(f"index_finger_{s}" for s in ("mcp_flex", "mcp_abd", "pip", "dip")),
    "middle": tuple(f"middle_finger_{s}" for s in ("mcp_flex", "mcp_abd", "pip", "dip")),
    "ring": tuple(f"ring_finger_{s}" for s in ("mcp_flex", "mcp_abd", "pip", "dip")),
    "pinky": tuple(f"pinky_{s}" for s in ("mcp_flex", "mcp_abd", "pip", "dip")),
}
# Beta 2 MJCF actuator codes (Shadow-hand style: TH, FF, MF, RF, LF; J0..J3).
_ACTUATOR_CODES = {"thumb": "TH", "index": "FF", "middle": "MF", "ring": "RF", "pinky": "LF"}
# Name of the fingertip site (MJCF) / link (URDF), without side prefix.
_TIP_NAMES = {
    "thumb": "thumb_tip",
    "index": "index_finger_tip",
    "middle": "middle_finger_tip",
    "ring": "ring_finger_tip",
    "pinky": "pinky_tip",
}


def side_prefix(side: Side) -> str:
    if side not in SIDES:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return "l_" if side == "left" else "r_"


def sdk_index(finger: str, joint: int) -> int:
    """SDK flat index of ``joint`` (1..4) of ``finger``."""
    if not 1 <= joint <= 4:
        raise ValueError(f"joint must be 1..4, got {joint}")
    return FINGERS.index(finger) * 4 + (joint - 1)


def model_joint_names(side: Side) -> tuple[str, ...]:
    """Beta 2 MJCF/URDF joint names, in SDK order."""
    p = side_prefix(side)
    return tuple(p + n for f in FINGERS for n in _MODEL_JOINTS[f])


def mjcf_actuator_names(side: Side) -> tuple[str, ...]:
    """Beta 2 MJCF actuator names, in SDK order."""
    p = side_prefix(side)
    return tuple(f"{p}{_ACTUATOR_CODES[f]}J{j}" for f in FINGERS for j in range(4))


def tip_names(side: Side) -> tuple[str, ...]:
    """Fingertip site (MJCF) / link (URDF) names, in finger order."""
    p = side_prefix(side)
    return tuple(p + _TIP_NAMES[f] for f in FINGERS)


def permutation(src: Sequence[str], dst: Sequence[str]) -> np.ndarray:
    """Index array ``perm`` with ``x_dst = x_src[..., perm]``.

    Raises unless ``src`` and ``dst`` are the same set of unique names.
    """
    src, dst = list(src), list(dst)
    if len(set(src)) != len(src) or len(set(dst)) != len(dst):
        raise ValueError("joint name lists must not contain duplicates")
    if set(src) != set(dst):
        missing, extra = sorted(set(dst) - set(src)), sorted(set(src) - set(dst))
        raise ValueError(f"joint name sets differ: missing from src {missing}, not in dst {extra}")
    lookup = {n: i for i, n in enumerate(src)}
    return np.array([lookup[n] for n in dst], dtype=np.intp)


def to_sdk(q: np.ndarray, names: Sequence[str], side: Side) -> np.ndarray:
    """Reorder ``q[..., len(names)]`` given in the ``names`` ordering into SDK order."""
    q = np.asarray(q)
    _check_last_dim(q, len(names))
    return q[..., permutation(names, model_joint_names(side))]


def from_sdk(q_sdk: np.ndarray, names: Sequence[str], side: Side) -> np.ndarray:
    """Reorder SDK-order ``q_sdk[..., 20]`` into the ``names`` ordering."""
    q_sdk = np.asarray(q_sdk)
    _check_last_dim(q_sdk, N_JOINTS)
    return q_sdk[..., permutation(model_joint_names(side), names)]


def _check_last_dim(q: np.ndarray, n: int) -> None:
    if q.ndim == 0 or q.shape[-1] != n:
        raise ValueError(f"expected last dimension {n}, got shape {q.shape}")
