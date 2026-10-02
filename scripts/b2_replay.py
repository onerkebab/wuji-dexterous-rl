"""B2: floating-hand replay video of a retargeted session (no robot, no physics).

Without RGB (stub): camera view (ZED pose and field of view, AS-01 intrinsics) | third-person view.
With RGB (real sessions; auto-found under data/raw/ego-clinical-task/unscrew_cap/<session>/rgb.mp4):
  RGB + A2's human keypoints projected with the session intrinsics (green, tips red)
  | RGB with the Wuji render composited from the same camera | third-person view.
Each Wuji hand is posed at its fitted palm pose (W1-B05). In the render views, small spheres mark the
retarget targets (W1-B08 direction targets, Wuji wrist frame): orange = fingertips, blue = other
joints. Invalid frames hide the hand. MuJoCo cameras have a centred principal point, so renders are
shifted by (cx - W/2, cy - H/2) before compositing.

Usage: MUJOCO_GL=egl uv run python scripts/b2_replay.py --session session_032 --exp-id 20260930-b2-session032
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from wuji_cap.common import derived_io, frames
from wuji_cap.common import joint_order as jo
from wuji_cap.common.provenance import REPO_ROOT, provenance
from wuji_cap.sim import hand_keypoints as hk
from wuji_cap.sim.render import composite, draw_keypoints, find_rgb, load_camera, principal_point_shift, put
from wuji_cap.sim.retarget import WujiRetargeter
from wuji_cap.sim.scene import ZED_FY_PX, ZED_H_PX, FloatingHands, add_spheres

W, H = 640, 360
BANNER = 70
TIP_RGBA = (1.0, 0.45, 0.1, 1.0)
KP_RGBA = (0.15, 0.4, 0.9, 1.0)


def target_points_world(retargeter: WujiRetargeter, kp_cam: np.ndarray, T_cam_from_palm: np.ndarray, scale: float,
                        q_prev: np.ndarray | None, T_wc: np.ndarray) -> np.ndarray:
    """The retarget targets of one frame (as built in WujiRetargeter.run) in the world frame."""
    kp_palm = scale * frames.apply_T(frames.inv_T(T_cam_from_palm), kp_cam)
    return frames.apply_T(T_wc @ T_cam_from_palm, retargeter.targets(kp_palm, q_prev))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="stub_session_032")
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    kp, kp_meta = derived_io.read_npz(REPO_ROOT / f"data/derived/keypoints/{args.session}.npz")
    rt, _ = derived_io.read_npz(REPO_ROOT / f"data/derived/retarget/{args.session}.npz")
    stub = bool(kp_meta.get("params", {}).get("stub", False))
    T = len(rt["timestamps_s"]) if args.max_frames is None else min(args.max_frames, len(rt["timestamps_s"]))
    fps = 1.0 / np.median(np.diff(rt["timestamps_s"]))
    T_wc = frames.T_world_from_cam()
    retargeters = {s: WujiRetargeter(s) for s in jo.SIDES}  # only for rebuilding the targets

    rgb_path, cam = (None, None) if stub else (find_rgb(args.session), load_camera(args.session))
    rgb_frames = None
    if rgb_path is not None:
        if not np.isclose(cam["fy"], ZED_FY_PX, rtol=1e-4) or cam["height"] != ZED_H_PX:
            raise ValueError(f"session intrinsics fy={cam['fy']} h={cam['height']} differ from the scene camera")
        dx, dy = principal_point_shift(cam, W)
        rgb_frames = imageio.get_reader(rgb_path)

    scene = FloatingHands(W, H)
    renderer = mujoco.Renderer(scene.model, H, W)
    cam_zed = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, "zed")
    # Third-person camera: look at the median palm position from the side and above.
    palms = np.concatenate([rt[f"{s}_T_cam_from_palm"][rt[f"{s}_valid"], :3, 3] for s in jo.SIDES])
    free = mujoco.MjvCamera()
    free.lookat[:] = frames.apply_T(T_wc, np.median(palms, 0) if len(palms) else np.array([0, 0, 0.5]))
    free.distance, free.azimuth, free.elevation = 0.75, 145.0, -30.0

    out_dir = REPO_ROOT / "results" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / f"{args.session}_floating_hands.mp4"
    n_panels = 2 if rgb_frames is None else 3
    stills = {}
    with imageio.get_writer(video, fps=fps, codec="libx264", quality=8, macro_block_size=1) as w:
        for t in range(T):
            targets = []
            for s in jo.SIDES:
                if rt[f"{s}_valid"][t]:
                    T_wp = T_wc @ rt[f"{s}_T_cam_from_palm"][t]
                    scene.set_hand(s, T_wp, rt[f"{s}_q_wuji"][t])
                    q_prev = rt[f"{s}_q_wuji"][t - 1] if t > 0 and rt[f"{s}_valid"][t - 1] else None
                    targets.append(target_points_world(retargeters[s], kp[f"{s}_kp21_cam"][t],
                                                       rt[f"{s}_T_cam_from_palm"][t], float(rt[f"{s}_scale"]),
                                                       q_prev, T_wc))
                else:
                    scene.set_hand(s, None, None)
            scene.forward()

            def render(camera, markers: bool) -> np.ndarray:
                renderer.update_scene(scene.data, camera)
                if markers:
                    for p in targets:
                        add_spheres(renderer.scene, p[list(hk.MP_TIPS)], TIP_RGBA)
                        add_spheres(renderer.scene, np.delete(p, list(hk.MP_TIPS), 0), KP_RGBA, 0.003)
                return renderer.render().copy()

            third = render(free, True)
            put(third, "third-person view", (10, 20))
            if rgb_frames is None:
                zed = render(cam_zed, True)
                put(zed, "camera view (ZED pose, AS-01)", (10, 20))
                tiles = [zed, third]
            else:
                rgb = cv2.resize(rgb_frames.get_data(int(rt["frame_index"][t])), (W, H), interpolation=cv2.INTER_AREA)
                human = rgb.copy()
                for s in jo.SIDES:
                    draw_keypoints(human, kp[f"{s}_kp21_cam"][t], cam)
                put(human, "RGB + human keypoints (A2)", (10, 20), color=(255, 255, 255))
                comp = composite(renderer, scene.data, cam_zed, rgb, dx, dy)
                put(comp, "RGB + Wuji render, same camera", (10, 20), color=(255, 255, 255))
                tiles = [human, comp, third]
            top = np.full((BANNER, n_panels * W, 3), 255, np.uint8)
            title = f"{args.session}  frame {int(rt['frame_index'][t])}  t={rt['timestamps_s'][t]:.2f}s  kinematic only (AS-12)"
            put(top, title + ("   STUB keypoints (Wuji FK, not human data)" if stub else ""), (10, 24), 0.6,
                (160, 30, 30) if stub else (20, 20, 20), 2)
            for j, s in enumerate(jo.SIDES):
                if rt[f"{s}_valid"][t]:
                    e = 1e3 * np.nanmax(rt[f"{s}_tip_err_m"][t])
                    msg = f"{s}: tip err max {e:4.1f} mm, at limit {int(rt[f'{s}_at_limit'][t].sum())}/20"
                else:
                    msg = f"{s}: INVALID (hidden)"
                put(top, msg, (10 + j * W, 54), 0.5)
            frame = np.vstack([top, np.hstack(tiles)])
            w.append_data(frame)
            if t in (0, T // 3, 2 * T // 3, T - 1):
                stills[t] = frame
    renderer.close()
    if rgb_frames is not None:
        rgb_frames.close()

    sheet = np.vstack([stills[k] for k in sorted(stills)])
    imageio.imwrite(out_dir / f"{args.session}_floating_hands_stills.png", cv2.resize(sheet, None, fx=0.6, fy=0.6))
    meta = {"video": str(video), "rgb": None if rgb_path is None else str(rgb_path), "n_frames": T, "fps": fps,
            "zed_fovy_deg": float(scene.model.cam_fovy[cam_zed]), "camera": cam,
            "provenance": provenance("scripts/b2_replay.py")}
    (out_dir / "replay.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {video}")


if __name__ == "__main__":
    main()
