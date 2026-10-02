"""MuJoCo render scenes. B2: both Wuji hands floating in the world frame (no robot), posed per frame.

World frame and camera convention: wuji_cap/common/frames.py. The camera sits at the world origin.
"""

from __future__ import annotations

import mujoco
import numpy as np

from wuji_cap.common import frames
from wuji_cap.common import joint_order as jo
from wuji_cap.sim import wuji_model as wm

# ZED rectified-left intrinsics (AS-01): fx = fy ~ 1504.45 px at 1920 x 1080.
ZED_FY_PX, ZED_H_PX = 1504.45, 1080
ZED_FOVY_DEG = float(np.degrees(2 * np.arctan(ZED_H_PX / 2 / ZED_FY_PX)))
HIDDEN_POS = np.array([0.0, 0.0, -10.0])


def mujoco_cam_xyaxes(R_world_from_optical: np.ndarray) -> np.ndarray:
    """MuJoCo camera ``xyaxes`` for an optical (x right, y down, z forward) camera orientation.

    A MuJoCo camera looks along its -z with +y up, so its x = optical x and its y = -optical y.
    """
    R = np.asarray(R_world_from_optical)
    return np.concatenate([R[:, 0], -R[:, 1]])


def build_floating_hands(width: int, height: int) -> mujoco.MjModel:
    """Both Beta 2 MJCFs as free-standing bodies (poses set via ``model.body_pos/quat``), a camera
    ``zed`` at the origin with the ZED field of view, lights, and a plain backdrop."""
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth, spec.visual.global_.offheight = width, height
    for side in jo.SIDES:
        child = mujoco.MjSpec.from_file(str(wm.mjcf_path(side)))
        # Adopt the hand's solver options so attach does not warn (and write MUJOCO_LOG.TXT).
        for opt in ("tolerance", "iterations", "integrator", "jacobian"):
            setattr(spec.option, opt, getattr(child.option, opt))
        spec.worldbody.add_frame().attach_body(child.body(jo.side_prefix(side) + "wrist"), "", "")
    spec.worldbody.add_camera(name="zed", pos=[0, 0, 0], fovy=ZED_FOVY_DEG,
                              xyaxes=mujoco_cam_xyaxes(frames.R_WORLD_FROM_CAM).tolist())
    spec.worldbody.add_light(pos=[0, 0, 1.0], dir=[0.3, 0, -1], diffuse=[0.6, 0.6, 0.6])
    spec.worldbody.add_light(pos=[0, 0.6, 0.4], dir=[0.4, -0.6, -0.3], diffuse=[0.4, 0.4, 0.4])
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.93, 0.94, 0.96],
                     rgb2=[0.75, 0.78, 0.82], width=64, height=64)
    return spec.compile()


class FloatingHands:
    def __init__(self, width: int, height: int):
        self.model = build_floating_hands(width, height)
        self.data = mujoco.MjData(self.model)
        self.hands = {s: wm.load_hand(s, self.model) for s in jo.SIDES}

    def set_hand(self, side: jo.Side, T_world_from_palm: np.ndarray | None, q_sdk: np.ndarray | None) -> None:
        """Pose one hand; ``None`` hides it (moved far below the scene)."""
        h = self.hands[side]
        if T_world_from_palm is None or q_sdk is None:
            self.model.body_pos[h.wrist_body_id] = HIDDEN_POS
            return
        self.model.body_pos[h.wrist_body_id] = T_world_from_palm[:3, 3]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, T_world_from_palm[:3, :3].ravel())
        self.model.body_quat[h.wrist_body_id] = q
        h.set_q_sdk(self.data, q_sdk)

    def forward(self) -> None:
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)  # camera poses; mj_kinematics leaves cam_xmat at zero


def add_spheres(scene: mujoco.MjvScene, points: np.ndarray, rgba, radius: float = 0.004) -> None:
    """Append marker spheres to an already updated scene."""
    for p in points:
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [radius] * 3,
                            np.asarray(p, float), np.eye(3).ravel(), np.asarray(rgba, np.float32))
        scene.ngeom += 1
