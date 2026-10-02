"""B5 before/after video: RGB + keypoints | G1 at the pre-B5 placement | G1 at the W1-A06 placement.

Both robot panels use one world-fixed camera behind the robots (robot left = image left), so the lowered
robot appears lower and the palm targets (green) sit at the same place in both. Red = achieved palm. Each robot
plays its own playback arrays (``*_cmd``, B4): ``--before`` is a retarget directory made before B5 (default
data/derived/retarget_pre_b5, a copy taken before the 20261001-b5-batch rerun); after = data/derived/retarget.

Usage: MUJOCO_GL=egl uv run python scripts/b5_compare_video.py --session session_032 --exp-id 20261001-b5-videos
Writes results/<exp_id>/<session>_b5_before_after.mp4 (gitignored), _stills.png and _compare.json.
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

from wuji_cap.common import derived_io
from wuji_cap.common import joint_order as jo
from wuji_cap.common.provenance import REPO_ROOT, provenance
from wuji_cap.sim import wuji_model as wm
from wuji_cap.sim.placement import placed_g1
from wuji_cap.sim.render import draw_keypoints, find_rgb, g1_behind_camera, load_camera, put
from wuji_cap.sim.scene import add_spheres

W, H = 640, 400
BANNER = 64
TARGET_RGBA, REACHED_RGBA = (0.1, 0.75, 0.2, 1.0), (0.9, 0.15, 0.15, 1.0)
RED = (200, 30, 30)


def robot_panel(g, renderer, cam, rt, t, title):
    g.data.qpos[:] = g.q_home
    for s in jo.SIDES:
        g.data.qpos[g.arm_qpos_adr[s]] = rt[f"{s}_q_g1_arm_cmd"][t]
        g.hands[s].set_q_sdk(g.data, rt[f"{s}_q_wuji_cmd"][t])
    mujoco.mj_kinematics(g.model, g.data)
    mujoco.mj_camlight(g.model, g.data)
    renderer.update_scene(g.data, cam)
    lines = []
    for s in jo.SIDES:
        if not rt[f"{s}_valid"][t]:
            lines.append((f"{s}: held (not usable)", (120, 120, 120), 1))
            continue
        add_spheres(renderer.scene, [rt[f"{s}_T_world_from_palm_target"][t][:3, 3]], TARGET_RGBA, 0.014)
        add_spheres(renderer.scene, [g.data.xpos[g.palm_body_ids[s]]], REACHED_RGBA, 0.009)
        fail = bool(rt[f"{s}_reach_failure"][t])
        lines.append((f"{s}: palm err {1e3 * rt[f'{s}_ik_pos_err_m'][t]:5.1f} mm" + ("  REACH FAIL" if fail else ""),
                      RED if fail else (20, 20, 20), 2 if fail else 1))
    img = renderer.render().copy()
    put(img, title, (10, 24), 0.6, thick=2)
    for k, (txt, col, th) in enumerate(lines):
        put(img, txt, (10, 50 + 22 * k), 0.5, col, th)
    put(img, "robot L", (10, H - 12), 0.5, (60, 60, 200), 2)
    put(img, "robot R", (W - 80, H - 12), 0.5, (60, 60, 200), 2)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="session_032")
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--before", default="data/derived/retarget_pre_b5")
    ap.add_argument("--frames", type=int, nargs=2, default=None, help="first and last row (inclusive)")
    args = ap.parse_args()

    before_path = REPO_ROOT / args.before / f"{args.session}.npz"
    after_path = REPO_ROOT / f"data/derived/retarget/{args.session}.npz"
    kp, _ = derived_io.read_npz(REPO_ROOT / f"data/derived/keypoints/{args.session}.npz")
    rts = {}
    for name, path in (("before", before_path), ("after", after_path)):
        rt, meta = derived_io.read_npz(path)
        got = meta.get("b3", {}).get("placement", {"name": "none"})["name"]
        if got != {"before": "none", "after": "b5"}[name]:
            raise ValueError(f"{path}: placement {got!r}, expected the {name} placement")
        rts[name] = rt
    if not np.array_equal(rts["before"]["frame_index"], rts["after"]["frame_index"]):
        raise ValueError("before and after retarget files have different rows")

    cam_meta, rgb_path = load_camera(args.session), find_rgb(args.session)
    reader = imageio.get_reader(rgb_path)
    robots = {"before": placed_g1(W, H, placed=False), "after": placed_g1(W, H, placed=True)}
    renderers = {k: mujoco.Renderer(g.model, H, W) for k, g in robots.items()}
    g1_behind_camera(robots["before"])  # (only to ghost the head mesh)
    view = g1_behind_camera(robots["after"], view={"distance": 1.45, "elevation": -45.0, "lookat_dz": 0.25})

    T = len(rts["after"]["frame_index"])
    t0, t1 = (0, T - 1) if args.frames is None else args.frames
    out_dir = REPO_ROOT / "results" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / f"{args.session}_b5_before_after.mp4"
    fps = 1.0 / np.median(np.diff(rts["after"]["timestamps_s"]))
    counts = {k: {s: int((rt[f"{s}_valid"] & rt[f"{s}_reach_failure"])[t0:t1 + 1].sum()) for s in jo.SIDES}
              for k, rt in rts.items()}
    worst = t0 + int(np.argmax(np.nan_to_num(rts["before"]["left_ik_pos_err_m"][t0:t1 + 1])
                               + np.nan_to_num(rts["before"]["right_ik_pos_err_m"][t0:t1 + 1])))
    still_ts = sorted({t0, worst, (t0 + t1) // 2, t1})
    stills = []
    with imageio.get_writer(video, fps=fps, codec="libx264", quality=8, macro_block_size=1) as w:
        for t in range(t0, t1 + 1):
            rgb = reader.get_data(int(rts["after"]["frame_index"][t]))
            rgb = cv2.resize(rgb, (W, int(round(rgb.shape[0] * W / rgb.shape[1]))), interpolation=cv2.INTER_AREA)
            for s in jo.SIDES:
                draw_keypoints(rgb, kp[f"{s}_kp21_cam"][t], cam_meta)  # keypoint rows = retarget rows
            human = np.full((H, W, 3), 255, np.uint8)
            y = H - rgb.shape[0]  # title strip above the image
            human[y:] = rgb[:H - y]
            put(human, "human (RGB + keypoints)", (10, 24), 0.6, thick=2)
            before = robot_panel(robots["before"], renderers["before"], view, rts["before"], t,
                                 "before: robot at home pose")
            after = robot_panel(robots["after"], renderers["after"], view, rts["after"], t,
                                "after: 2 cm closer, 31.5 cm lower (W1-A06)")
            top = np.full((BANNER, 3 * W, 3), 255, np.uint8)
            put(top, f"{args.session}  frame {int(rts['after']['frame_index'][t])}  t={rts['after']['timestamps_s'][t]:.2f}s"
                     "   green = palm target (fixed in the world), red = robot palm", (10, 26), 0.6, thick=2)
            put(top, f"reach failures in this clip (L/R): before {counts['before']['left']}/{counts['before']['right']}"
                     f", after {counts['after']['left']}/{counts['after']['right']}.  Kinematic only (AS-12); "
                     "camera AS-01, wrist mount AS-02", (10, 52), 0.5)
            frame = np.vstack([top, np.hstack([human, before, after])])
            w.append_data(frame)
            if t in still_ts:
                stills.append(frame)
    reader.close()
    for r in renderers.values():
        r.close()
    imageio.imwrite(out_dir / f"{args.session}_b5_before_after_stills.png",
                    cv2.resize(np.vstack(stills), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    meta = {"video": str(video), "before": str(before_path), "after": str(after_path), "rows": [t0, t1],
            "fps": fps, "still_rows": still_ts, "reach_failures_in_clip": counts,
            "before_sha256": wm.sha256(before_path), "after_sha256": wm.sha256(after_path),
            "assumptions": ["AS-01", "AS-02", "AS-09", "AS-12"], "decisions": ["W1-B16", "W1-A06"],
            "provenance": provenance("scripts/b5_compare_video.py")}
    (out_dir / f"{args.session}_b5_compare.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {video}")


if __name__ == "__main__":
    main()
