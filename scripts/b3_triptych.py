"""DoD 3: triptych video of one session: RGB + human keypoints | RGB + floating Wuji hands | G1 + Wuji.

Panel 1: the session RGB with A's MediaPipe-21 keypoints projected (session intrinsics).
Panel 2: the B2 result: both Wuji hands at their fitted palm poses (W1-B05) rendered from the ZED camera
         (AS-01 intrinsics) and blended onto the RGB.
Panel 3: the B3 result: the G1 with both Wuji hands, arms from the IK (W1-B10), fixed pelvis/legs/waist
         (AS-09), seen from behind and above (robot left = image left, as in panel 1; head drawn faint so it does not
         hide the hands); green spheres = palm targets. Inset: the G1 head-camera view (AS-01).
         A side with a reach failure (W1-B11) is flagged in red; its arm shows the best-effort pose.
B4 (W1-B14, W1-B15): panel 3 plays the ``*_cmd`` arrays: a hand out of view (or not retargeted) holds its last
pose (home before the first one), labelled "held" in grey, without a palm target; after a gap it returns at a capped
joint speed, labelled "returning". Panel 2 shows only usable frames.
Frames are written one per keypoint row (frame_index -> rgb.mp4 frame) at the median frame rate; the source
timestamps are not uniform (AS-A05), so the video's clock is approximate. Kinematic only (AS-12).

Usage: MUJOCO_GL=egl uv run python scripts/b3_triptych.py --session session_032 --exp-id 20261001-b3-triptych
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from wuji_cap.common import derived_io, frames
from wuji_cap.common import joint_order as jo
from wuji_cap.common.provenance import REPO_ROOT, provenance
from wuji_cap.sim import g1_model
from wuji_cap.sim.placement import move_camera_to_reference, placed_g1
from wuji_cap.sim.render import (composite, draw_keypoints, find_rgb, g1_behind_camera, load_camera,
                                  principal_point_shift, put)
from wuji_cap.sim.scene import ZED_FY_PX, ZED_H_PX, FloatingHands, add_spheres

W, H = 640, 360
INSET_W, INSET_H = 224, 126
BANNER = 74
TARGET_RGBA = (0.1, 0.75, 0.2, 1.0)
RED = (200, 30, 30)
GREY = (120, 120, 120)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="session_032")
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    kp_path = REPO_ROOT / f"data/derived/keypoints/{args.session}.npz"
    rt_path = REPO_ROOT / f"data/derived/retarget/{args.session}.npz"
    kp, _ = derived_io.read_npz(kp_path)
    rt, rt_meta = derived_io.read_npz(rt_path)
    derived_io.check_retarget(rt)
    if "b3" not in rt_meta:
        raise ValueError(f"{rt_path} has no B3 arm fields; run scripts/b3_arm_ik.py first")
    T = len(rt["timestamps_s"]) if args.max_frames is None else min(args.max_frames, len(rt["timestamps_s"]))
    fps = 1.0 / np.median(np.diff(rt["timestamps_s"]))

    rgb_path, cam = find_rgb(args.session), load_camera(args.session)
    if rgb_path is None or cam is None:
        raise FileNotFoundError(f"RGB or metadata for {args.session} not found under data/raw")
    if not np.isclose(cam["fy"], ZED_FY_PX, rtol=1e-4) or cam["height"] != ZED_H_PX:
        raise ValueError(f"session intrinsics fy={cam['fy']} h={cam['height']} differ from the scene camera")
    dx, dy = principal_point_shift(cam, W)
    dxi, dyi = principal_point_shift(cam, INSET_W)
    reader = imageio.get_reader(rgb_path)

    floating = FloatingHands(W, H)
    r_float = mujoco.Renderer(floating.model, H, W)
    cam_zed = mujoco.mj_name2id(floating.model, mujoco.mjtObj.mjOBJ_CAMERA, "zed")
    T_wc_float = frames.T_world_from_cam()

    placement = rt_meta["b3"].get("placement", {"name": "none"})  # render the robot the IK solved for
    g = placed_g1(W, H, placement["name"] == "b5")
    move_camera_to_reference(g, g1_model.HEAD_CAMERA)  # inset = the human's viewpoint (AS-01)
    r_g1 = mujoco.Renderer(g.model, H, W)
    r_inset = mujoco.Renderer(g.model, INSET_H, INSET_W)
    head = mujoco.mj_name2id(g.model, mujoco.mjtObj.mjOBJ_CAMERA, g1_model.HEAD_CAMERA)
    behind = g1_behind_camera(g, right_shift_m=0.13)  # robot left of centre, clear of the head-camera inset

    out_dir = REPO_ROOT / "results" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / f"{args.session}_triptych.mp4"
    fail_any = rt["left_reach_failure"] | rt["right_reach_failure"]
    held_any = rt["left_held"] | rt["right_held"]
    ret_rows = np.flatnonzero((rt["left_arm_catching_up"] | rt["right_arm_catching_up"])[:T])
    still_ts = sorted({0, T // 3, 2 * T // 3, T - 1} | ({int(np.flatnonzero(fail_any[:T])[len(np.flatnonzero(fail_any[:T])) // 2])}
                                                         if fail_any[:T].any() else set())
                      | ({int(np.flatnonzero(held_any[:T])[len(np.flatnonzero(held_any[:T])) // 2])} if held_any[:T].any() else set())
                      | ({int(ret_rows[len(ret_rows) // 2])} if ret_rows.size else set()))
    stills = {}
    with imageio.get_writer(video, fps=fps, codec="libx264", quality=8, macro_block_size=1) as w:
        for t in range(T):
            rgb = cv2.resize(reader.get_data(int(rt["frame_index"][t])), (W, H), interpolation=cv2.INTER_AREA)

            human = rgb.copy()
            for s in jo.SIDES:
                draw_keypoints(human, kp[f"{s}_kp21_cam"][t], cam)
            put(human, "1  RGB + human keypoints (A)", (10, 22), 0.55, (255, 255, 255), 2)

            for s in jo.SIDES:
                ok = bool(rt[f"{s}_valid"][t])
                floating.set_hand(s, T_wc_float @ rt[f"{s}_T_cam_from_palm"][t] if ok else None,
                                  rt[f"{s}_q_wuji"][t] if ok else None)
            floating.forward()
            hands = composite(r_float, floating.data, cam_zed, rgb, dx, dy)
            put(hands, "2  floating Wuji hands (B2)", (10, 22), 0.55, (255, 255, 255), 2)

            g.data.qpos[:] = g.q_home
            targets = []
            for s in jo.SIDES:  # playback arrays (B4): held sides keep their last usable pose
                g.data.qpos[g.arm_qpos_adr[s]] = rt[f"{s}_q_g1_arm_cmd"][t]
                g.hands[s].set_q_sdk(g.data, rt[f"{s}_q_wuji_cmd"][t])
                if rt[f"{s}_valid"][t]:
                    targets.append(rt[f"{s}_T_world_from_palm_target"][t][:3, 3])
            mujoco.mj_kinematics(g.model, g.data)
            mujoco.mj_camlight(g.model, g.data)
            r_g1.update_scene(g.data, behind)
            add_spheres(r_g1.scene, targets, TARGET_RGBA, 0.012)
            robot = r_g1.render().copy()
            inset = composite(r_inset, g.data, head, cv2.resize(rgb, (INSET_W, INSET_H), interpolation=cv2.INTER_AREA),
                              dxi, dyi)
            robot[H - INSET_H - 6:H - 6, W - INSET_W - 6:W - 6] = inset
            cv2.rectangle(robot, (W - INSET_W - 7, H - INSET_H - 7), (W - 6, H - 6), (40, 40, 40), 1)
            put(robot, "original head camera", (W - INSET_W - 2, H - INSET_H - 12), 0.4)
            put(robot, "3  G1 + Wuji from behind (B3); green = palm target", (10, 22), 0.55)
            put(robot, "robot L", (10, 50), 0.55, (60, 60, 200), 2)
            put(robot, "robot R", (W - 90, 50), 0.55, (60, 60, 200), 2)

            top = np.full((BANNER, 3 * W, 3), 255, np.uint8)
            put(top, f"{args.session}  frame {int(rt['frame_index'][t])}  t={rt['timestamps_s'][t]:.2f}s   "
                     "kinematic only (AS-12); camera AS-01, wrist mount AS-02, fixed waist AS-09; "
                     + ("robot 2 cm closer, 31.5 cm lower (W1-A06)" if placement["name"] == "b5" else "pre-B5 placement"), (10, 24), 0.6, thick=2)
            for j, s in enumerate(jo.SIDES):
                if not rt[f"{s}_valid"][t]:
                    why = "out of view" if not rt[f"{s}_visible"][t] else "not retargeted"
                    put(top, f"{s}: {why}", (10 + W + j * W // 2, 56), 0.45, GREY)
                    put(top, f"{s}: held ({why})", (10 + 2 * W + j * W // 2, 56), 0.45, GREY)
                    continue
                e = 1e3 * np.nanmax(rt[f"{s}_tip_err_m"][t])
                put(top, f"{s}: tip err max {e:4.1f} mm, at limit {int(rt[f'{s}_at_limit'][t].sum())}/20",
                    (10 + W + j * W // 2, 56), 0.45)
                pe, re = 1e3 * rt[f"{s}_ik_pos_err_m"][t], np.degrees(rt[f"{s}_ik_rot_err_rad"][t])
                fail = bool(rt[f"{s}_reach_failure"][t])
                ret = bool(rt[f"{s}_arm_catching_up"][t] or rt[f"{s}_wuji_catching_up"][t])
                put(top, f"{s}: IK {pe:5.1f} mm {re:4.1f} deg" + ("  REACH FAIL" if fail else ""),
                    (10 + 2 * W + j * W // 2, 56), 0.45, RED if fail else (20, 20, 20), 2 if fail else 1)
                if ret:
                    put(top, "returning", (10 + 2 * W + j * W // 2, 70), 0.4, GREY)
            frame = np.vstack([top, np.hstack([human, hands, robot])])
            w.append_data(frame)
            if t in still_ts:
                stills[t] = frame
    for r in (r_float, r_g1, r_inset):
        r.close()
    reader.close()

    sheet = np.vstack([stills[k] for k in sorted(stills)])
    imageio.imwrite(out_dir / f"{args.session}_triptych_stills.png", cv2.resize(sheet, None, fx=0.5, fy=0.5,
                                                                               interpolation=cv2.INTER_AREA))
    meta = {"video": str(video), "rgb": str(rgb_path), "keypoints": str(kp_path), "retarget": str(rt_path),
            "n_frames": T, "fps": fps, "still_rows": still_ts, "camera": cam,
            "assumptions": ["AS-01", "AS-02", "AS-09", "AS-12", "AS-A05"], "decisions": ["W1-B14", "W1-B15", "W1-A06"], "placement": placement,
            "provenance": provenance("scripts/b3_triptych.py")}
    (out_dir / f"{args.session}_triptych.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {video}")


if __name__ == "__main__":
    main()
