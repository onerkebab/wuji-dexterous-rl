"""Wuji Hand 2 (Beta 2) model loading for MuJoCo, with SDK-order joint access.

Model source: ``../wuji-description/hand2/hand2_beta2/body/{mjcf,urdf}/{left,right}.*`` pinned at
commit c2cd7f8 (DECISIONS W1-06). Override the checkout location with ``WUJI_DESCRIPTION_DIR``.
All joint access goes through names resolved with ``mj_name2id``; the file order is never assumed.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from wuji_cap.common import joint_order as jo
from wuji_cap.common.joint_order import Side

REPO_ROOT = Path(__file__).resolve().parents[2]
WUJI_DESCRIPTION_COMMIT = "c2cd7f8"
_BETA2_BODY = Path("hand2/hand2_beta2/body")


def wuji_description_dir() -> Path:
    return Path(os.environ.get("WUJI_DESCRIPTION_DIR", REPO_ROOT.parent / "wuji-description"))


def mjcf_path(side: Side) -> Path:
    return wuji_description_dir() / _BETA2_BODY / "mjcf" / f"{side}.xml"


def urdf_path(side: Side) -> Path:
    return wuji_description_dir() / _BETA2_BODY / "urdf" / f"{side}.urdf"


def wuji_description_commit() -> str:
    """Full commit hash of the wuji-description checkout (plus '-dirty' if modified)."""
    d = wuji_description_dir()
    head = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    dirty = subprocess.run(["git", "-C", d, "status", "--porcelain"], capture_output=True, text=True, check=True)
    return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class WujiHand:
    """A single Wuji hand MjModel with SDK-order index arrays."""

    side: Side
    model: mujoco.MjModel
    qpos_adr: np.ndarray  # [20] qpos address of SDK joint i
    jnt_ids: np.ndarray  # [20] joint id of SDK joint i
    act_ids: np.ndarray  # [20] actuator id of SDK joint i
    tip_site_ids: np.ndarray  # [5] fingertip site ids, finger order
    wrist_body_id: int

    @property
    def joint_range(self) -> np.ndarray:
        """[20, 2] joint limits (rad), SDK order."""
        return self.model.jnt_range[self.jnt_ids].copy()

    def set_q_sdk(self, data: mujoco.MjData, q_sdk: np.ndarray) -> None:
        """Write SDK-order joint angles into ``data.qpos`` and ``data.ctrl``."""
        q_sdk = np.asarray(q_sdk, dtype=float)
        if q_sdk.shape != (jo.N_JOINTS,):
            raise ValueError(f"expected shape (20,), got {q_sdk.shape}")
        data.qpos[self.qpos_adr] = q_sdk
        data.ctrl[self.act_ids] = q_sdk

    def get_q_sdk(self, data: mujoco.MjData) -> np.ndarray:
        return data.qpos[self.qpos_adr].copy()

    def tips_in_wrist(self, data: mujoco.MjData) -> np.ndarray:
        """[5, 3] fingertip site positions in the wrist body frame (needs mj_kinematics first)."""
        R = data.xmat[self.wrist_body_id].reshape(3, 3)
        p = data.xpos[self.wrist_body_id]
        return (data.site_xpos[self.tip_site_ids] - p) @ R


def _id(model: mujoco.MjModel, obj: mujoco.mjtObj, name: str) -> int:
    i = mujoco.mj_name2id(model, obj, name)
    if i < 0:
        raise KeyError(f"{obj.name} {name!r} not found in model")
    return i


def load_hand(side: Side, model: mujoco.MjModel | None = None) -> WujiHand:
    """Load the Beta 2 MJCF for ``side`` (or index into an already built ``model``)."""
    if model is None:
        model = mujoco.MjModel.from_xml_path(str(mjcf_path(side)))
    J, A = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR
    jnt_ids = np.array([_id(model, J, n) for n in jo.model_joint_names(side)])
    act_ids = np.array([_id(model, A, n) for n in jo.mjcf_actuator_names(side)])
    for j, a in zip(jnt_ids, act_ids):
        if model.actuator_trnid[a, 0] != j:
            raise ValueError(f"actuator {model.actuator(a).name} does not drive joint {model.joint(j).name}")
    if not np.all(model.jnt_type[jnt_ids] == mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("expected 20 hinge joints")
    return WujiHand(
        side=side,
        model=model,
        qpos_adr=model.jnt_qposadr[jnt_ids].copy(),
        jnt_ids=jnt_ids,
        act_ids=act_ids,
        tip_site_ids=np.array([_id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in jo.tip_names(side)]),
        wrist_body_id=_id(model, mujoco.mjtObj.mjOBJ_BODY, jo.side_prefix(side) + "wrist"),
    )


def mjcf_joint_names_in_file_order(model: mujoco.MjModel) -> list[str]:
    return [model.joint(i).name for i in range(model.njnt)]
