"""G1 (29 DoF) with both Wuji Beta 2 hands on its wrists, for B3 arm IK and rendering.

Composition ported from TWIST2 ``deploy_real/wuji_mujoco_model.py`` (branch dev/wuji-hand @ cf391dd,
``compose_g1_wuji_model``): attach each Wuji MJCF to ``{side}_wrist_yaw_link`` through a mount frame,
after deleting the G1's ``{side}_rubber_hand`` placeholder, with an empty prefix so the official
``l_*``/``r_*`` names survive. Differences: the hand MJCFs come from ``wuji_model.mjcf_path`` (W1-06)
instead of retargeter YAMLs, and the result is wrapped with SDK-order hand access and G1 arm indices.

Mount (AS-02): TWIST2's visualization/test mount, not a dimensional specification.
G1 model (W1-07): ``../TWIST2/assets/g1/g1_sim2sim_29dof.xml``; override with ``TWIST2_DIR``.
Pose (W1-07, AS-09): the file's ``home`` keyframe (nominal standing); in IK only the two arms move.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

from wuji_cap.common import frames
from wuji_cap.common import joint_order as jo
from wuji_cap.common.joint_order import Side
from wuji_cap.common.provenance import REPO_ROOT
from wuji_cap.sim import wuji_model as wm
from wuji_cap.sim.scene import ZED_FOVY_DEG, mujoco_cam_xyaxes

MOUNT_POS = {"left": (0.0415, 0.003, 0.0), "right": (0.0415, -0.003, 0.0)}  # AS-02
MOUNT_QUAT_WXYZ = (float(np.sqrt(0.5)), 0.0, -float(np.sqrt(0.5)), 0.0)  # AS-02: Wuji -Z fingers -> wrist +X
G1_ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
TORSO_BODY = "torso_link"
HEAD_CAMERA = "head_zed"  # AS-01: ZED rectified-left at the D435 mount, ZED field of view


def twist2_dir() -> Path:
    return Path(os.environ.get("TWIST2_DIR", REPO_ROOT.parent / "TWIST2"))


def g1_xml_path() -> Path:
    return twist2_dir() / "assets/g1/g1_sim2sim_29dof.xml"


def arm_joint_names(side: Side) -> tuple[str, ...]:
    if side not in jo.SIDES:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return tuple(f"{side}_{j}_joint" for j in G1_ARM_JOINTS)


def build_spec(width: int = 640, height: int = 480) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(g1_xml_path()))
    spec.copy_during_attach = True
    spec.visual.global_.offwidth, spec.visual.global_.offheight = width, height
    for side in jo.SIDES:
        wrist = spec.body(f"{side}_wrist_yaw_link")
        if wrist is None:
            raise ValueError(f"G1 model has no {side}_wrist_yaw_link body")
        placeholder = spec.body(f"{side}_rubber_hand")
        if placeholder is not None:
            spec.delete(placeholder)
        child = mujoco.MjSpec.from_file(str(wm.mjcf_path(side)))
        # One global option block: keep the G1's settings (as TWIST2 does).
        for name in ("timestep", "tolerance", "integrator", "jacobian", "solver"):
            setattr(child.option, name, getattr(spec.option, name))
        frame = wrist.add_frame()
        frame.name = f"wuji_{side}_mount"
        frame.pos = MOUNT_POS[side]
        frame.quat = MOUNT_QUAT_WXYZ
        spec.attach(child, prefix="", frame=frame)
    T = frames.T_torso_from_cam()
    spec.body(TORSO_BODY).add_camera(name=HEAD_CAMERA, pos=T[:3, 3].tolist(), fovy=ZED_FOVY_DEG,
                                     xyaxes=mujoco_cam_xyaxes(T[:3, :3]).tolist())
    return spec


class G1Wuji:
    """Compiled G1 + 2 Wuji model with index arrays. ``palm`` = the Wuji wrist body (``l_wrist``/``r_wrist``),
    the same frame as ``{side}_T_cam_from_palm`` in the retarget files."""

    def __init__(self, width: int = 640, height: int = 480, pelvis_offset=(0.0, 0.0, 0.0)):
        self.pelvis_offset = np.asarray(pelvis_offset, dtype=float).copy()
        if self.pelvis_offset.shape != (3,) or not np.isfinite(self.pelvis_offset).all():
            raise ValueError("pelvis_offset must be three finite world-frame metres")
        self.model = build_spec(width, height).compile()
        self.data = mujoco.MjData(self.model)
        expected_nu = 29 + 2 * jo.N_JOINTS
        if self.model.nu != expected_nu:
            raise ValueError(f"expected {expected_nu} actuators (29 G1 + 20 per hand), found {self.model.nu}")
        self.hands = {s: wm.load_hand(s, self.model) for s in jo.SIDES}
        J = mujoco.mjtObj.mjOBJ_JOINT
        self.arm_jnt_ids = {s: np.array([wm._id(self.model, J, n) for n in arm_joint_names(s)]) for s in jo.SIDES}
        self.arm_qpos_adr = {s: self.model.jnt_qposadr[j].copy() for s, j in self.arm_jnt_ids.items()}
        self.arm_dof_adr = {s: self.model.jnt_dofadr[j].copy() for s, j in self.arm_jnt_ids.items()}
        waist = np.array([wm._id(self.model, J, n) for n in WAIST_JOINTS])
        self.waist_qpos_adr = self.model.jnt_qposadr[waist].copy()
        self.waist_dof_adr = self.model.jnt_dofadr[waist].copy()
        self.palm_body_ids = {s: self.hands[s].wrist_body_id for s in jo.SIDES}
        self.torso_body_id = wm._id(self.model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY)
        self.q_home = self.home_qpos()
        self.data.qpos[:] = self.q_home
        mujoco.mj_kinematics(self.model, self.data)

    def home_qpos(self) -> np.ndarray:
        """``home`` keyframe of the G1 file for the G1 joints (by name), Wuji joints at 0."""
        g1 = mujoco.MjModel.from_xml_path(str(g1_xml_path()))
        key = mujoco.mj_name2id(g1, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key < 0:
            raise KeyError("G1 model has no 'home' keyframe")
        q = self.model.qpos0.copy()
        for j in range(g1.njnt):
            name = g1.joint(j).name
            n = {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4}.get(int(g1.jnt_type[j]), 1)
            a, b = g1.jnt_qposadr[j], self.model.jnt_qposadr[wm._id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            q[b:b + n] = g1.key_qpos[key, a:a + n]
        pelvis = self.model.joint("pelvis").qposadr[0]
        q[pelvis:pelvis + 3] += self.pelvis_offset
        return q

    def frozen_dofs(self) -> np.ndarray:
        """All DoF indices except the 14 arm joints (pelvis, legs, waist, Wuji fingers)."""
        arm = np.concatenate([self.arm_dof_adr[s] for s in jo.SIDES])
        return np.setdiff1d(np.arange(self.model.nv), arm)

    def body_T(self, body_id: int, data: mujoco.MjData | None = None) -> np.ndarray:
        d = self.data if data is None else data
        T = np.eye(4)
        T[:3, :3] = d.xmat[body_id].reshape(3, 3)
        T[:3, 3] = d.xpos[body_id]
        return T

    def T_world_from_torso(self) -> np.ndarray:
        """Torso pose at the home pose. Fixed: pelvis, legs and waist never move (AS-09)."""
        d = mujoco.MjData(self.model)
        d.qpos[:] = self.q_home
        mujoco.mj_kinematics(self.model, d)
        return self.body_T(self.torso_body_id, d)
