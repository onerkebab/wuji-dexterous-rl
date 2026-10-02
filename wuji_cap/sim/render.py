"""Shared helpers for the replay and triptych videos (B2 replay, B3 triptych): RGB loading, keypoint
overlay, and compositing a MuJoCo render onto the RGB frame.

MuJoCo cameras have a centred principal point, so renders are shifted by (cx - W/2, cy - H/2) (scaled to the
output size) before compositing.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from wuji_cap.common.provenance import REPO_ROOT
from wuji_cap.sim import hand_keypoints as hk

WUJI_ALPHA = 0.65


def put(img, text, org, scale=0.5, color=(20, 20, 20), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def find_rgb(session: str) -> Path | None:
    p = REPO_ROOT / f"data/raw/ego-clinical-task/unscrew_cap/{session}/rgb.mp4"
    return p if p.exists() else None


def load_camera(session: str) -> dict | None:
    p = REPO_ROOT / f"data/raw/ego-clinical-task/unscrew_cap/{session}/metadata.json"
    return json.loads(p.read_text())["camera"] if p.exists() else None


def principal_point_shift(cam: dict, width: int) -> tuple[float, float]:
    k = width / cam["width"]
    return (cam["cx"] - cam["width"] / 2) * k, (cam["cy"] - cam["height"] / 2) * k


def draw_keypoints(img: np.ndarray, kp21_cam: np.ndarray, cam: dict) -> None:
    """Project MediaPipe-21 camera-frame keypoints into ``img`` (resized RGB): green chains, red tips."""
    if not np.isfinite(kp21_cam).all():
        return
    k = img.shape[1] / cam["width"]
    x = kp21_cam
    u = (cam["fx"] * x[:, 0] / x[:, 2] + cam["cx"]) * k
    v = (cam["fy"] * x[:, 1] / x[:, 2] + cam["cy"]) * k
    for c in hk.finger_chains():
        pts = np.stack([u[[0] + c], v[[0] + c]], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, (40, 220, 40), 1, cv2.LINE_AA)
    for i in range(21):
        cv2.circle(img, (int(u[i]), int(v[i])), 3 if i in hk.MP_TIPS else 2,
                   (230, 30, 30) if i in hk.MP_TIPS else (40, 220, 40), -1, cv2.LINE_AA)


def composite(renderer: mujoco.Renderer, data: mujoco.MjData, camera, rgb: np.ndarray, dx: float, dy: float,
              alpha: float = WUJI_ALPHA) -> np.ndarray:
    """Blend the render of ``camera`` onto ``rgb`` wherever the render hits geometry."""
    renderer.update_scene(data, camera)
    color = shift(renderer.render().copy(), dx, dy)
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera)
    mask = shift((renderer.render()[..., 0] >= 0).astype(np.uint8), dx, dy).astype(bool)
    renderer.disable_segmentation_rendering()
    out = rgb.copy()
    out[mask] = (alpha * color[mask] + (1 - alpha) * rgb[mask]).astype(np.uint8)
    return out


HEAD_MESH = "head_link"
HEAD_ALPHA = 0.12


PLACED_VIEW = {"distance": 0.9, "elevation": -58.0, "lookat_dz": -0.04}  # pre-B5 robot (B4)
PLACED_VIEW_B5 = {"distance": 1.2, "elevation": -50.0, "lookat_dz": 0.10}  # W1-A06 robot: hands sit higher on its body


def g1_behind_camera(g, ghost_head: bool = True, right_shift_m: float = 0.0, view: dict | None = None) -> mujoco.MjvCamera:
    """Free camera behind and above the G1, looking forward over its head at the hands, so robot left = image
    left as in the head camera and the RGB (B4). ``ghost_head`` makes the head mesh faint in ``g.model`` (render-only
    models) so it does not hide the hands. ``right_shift_m`` aims the camera that far toward the robot's right, which
    moves the robot left in the image (room for an inset on the right)."""
    m = g.model
    if ghost_head:
        for i in range(m.ngeom):
            if m.geom_dataid[i] >= 0 and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and m.mesh(m.geom_dataid[i]).name == HEAD_MESH:
                m.geom_rgba[i, 3] = HEAD_ALPHA
                m.geom_matid[i] = -1  # rgba applies only without a material
    if view is None:
        view = PLACED_VIEW_B5 if np.any(getattr(g, "pelvis_offset", 0)) else PLACED_VIEW
    cam = mujoco.MjvCamera()
    cam.lookat[:] = g.T_world_from_torso()[:3, 3] + [0.3, -right_shift_m, view["lookat_dz"]]
    cam.distance, cam.azimuth, cam.elevation = view["distance"], 0.0, view["elevation"]  # azimuth 0: along the robot's +x
    return cam
