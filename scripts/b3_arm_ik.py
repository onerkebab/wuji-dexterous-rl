"""B3: put both Wuji hands on the G1 and solve the arm IK for one retargeted session.

Reads  data/derived/retarget/<session>.npz   (B2 output: {side}_T_cam_from_palm, {side}_q_wuji, {side}_valid)
Writes the same file with the contract-v2 arm fields filled: {side}_q_g1_arm, {side}_ik_pos_err_m,
       {side}_ik_rot_err_rad; extra keys {side}_T_world_from_palm_target, {side}_ik_start_used,
       {side}_ik_n_iters, {side}_arm_at_limit, {side}_reach_failure (W1-B11), and the AS-B08 free-waist
       diagnostic {side}_ik_{pos_err_m,rot_err_rad}_free_waist_diag, q_waist_free_waist_diag [T, 3]
       (ArmIK.run_free_waist; not the robot motion); B4 (W1-B14, W1-B15) playback {side}_q_g1_arm_cmd [T, 7]
       (hold.playback: held pose through gaps, home before the first valid frame, return at <= 2 rad/s) and
       {side}_arm_catching_up [T]. B2's meta is kept; B3's goes under meta["b3"].
       results/<exp_id>/metrics.json      error percentiles + histograms, candidate reach-failure thresholds
                                          (W1-17), arm at-limit fractions, failure segments, target reach;
                                          all over valid (usable, B4) frames; held frames and playback speeds
       results/<exp_id>/ik_error_hist.png the W1-17 histograms
       results/<exp_id>/g1_stills.png     G1 + Wuji (playback arrays) at sample frames: head camera (AS-01) |
                                          view from behind (robot left = image left); green spheres = palm
                                          targets, red = palm reached

Kinematic only (AS-12). Depends on AS-01 (camera mount), AS-02 (Wuji mount), AS-09 (fixed pelvis/legs/waist).

Usage: MUJOCO_GL=egl uv run python scripts/b3_arm_ik.py --session session_032 --exp-id 20261001-b3-session032
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
from wuji_cap.sim import g1_model, hold
from wuji_cap.sim import wuji_model as wm
from wuji_cap.sim.arm_ik import POS_TOL_M, ROT_TOL_RAD, ArmIK, reach_failure
from wuji_cap.sim.placement import (LEG_PITCH_RAD, PELVIS_OFFSET_M, move_camera_to_reference, placed_g1,
                                    reference_T_world_from_cam)
from wuji_cap.sim.render import g1_behind_camera
from wuji_cap.sim.scene import add_spheres

POS_BINS_MM = np.logspace(-3, 3, 25)  # 0.001 mm .. 1 m, 4 bins per decade
ROT_BINS_DEG = np.logspace(-3, 2.5, 23)
POS_CANDIDATES_MM = (1, 5, 10, 20, 50)
ROT_CANDIDATES_DEG = (1, 5, 10, 20)
W, H = 480, 360
TARGET_RGBA, REACHED_RGBA = (0.1, 0.75, 0.2, 1.0), (0.9, 0.15, 0.15, 1.0)


def pct(x: np.ndarray, qs=(50, 90, 95, 99, 100)) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {"mean": float(x.mean()) if x.size else None} | {f"p{q}": (float(np.percentile(x, q)) if x.size else None) for q in qs}


def segments(mask: np.ndarray, frame_index: np.ndarray) -> list[list[int]]:
    """Contiguous runs of True as [first_frame, last_frame] (frame_index values)."""
    out, start = [], None
    for t, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = t
        elif not m and start is not None:
            out.append([int(frame_index[start]), int(frame_index[t - 1])])
            start = None
    return out


def hold_metrics(rt: dict, res, side: str) -> dict:
    """B4: held frames and the arm playback speeds (catch-up after gaps, rad/s)."""
    held = res.held[side]
    if not np.array_equal(held, ~rt[f"{side}_valid"]):
        raise ValueError(f"{side}: IK held frames differ from B2's invalid frames")
    return {"n_held": int(held.sum()), "n_not_visible": int((~rt[f"{side}_visible"]).sum()),
            "speed_cap_rad_s": hold.ARM_SPEED_CAP_RAD_S,
            "q_g1_arm_cmd_playback": hold.playback_stats(rt[f"{side}_q_g1_arm_cmd"], held, rt[f"{side}_arm_catching_up"],
                                                         rt["timestamps_s"])}


def side_metrics(g: g1_model.G1Wuji, side: str, res, frame_index: np.ndarray) -> dict:
    v = np.isfinite(res.pos_err_m[side])
    pos_mm, rot_deg = res.pos_err_m[side][v] * 1e3, np.degrees(res.rot_err_rad[side][v])
    d = mujoco.MjData(g.model)
    d.qpos[:] = g.q_home
    mujoco.mj_kinematics(g.model, d)
    shoulder = d.xpos[mujoco.mj_name2id(g.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_shoulder_pitch_link")]
    reach = np.linalg.norm(res.T_world_from_palm_target[side][v, :3, 3] - shoulder, axis=1)
    conv = (pos_mm < POS_TOL_M * 1e3) & (rot_deg < np.degrees(ROT_TOL_RAD))
    names = g1_model.arm_joint_names(side)
    return {
        "n_frames": int(len(v)), "n_solved": int(v.sum()),
        "pos_err_mm": pct(pos_mm), "rot_err_deg": pct(rot_deg),
        "pos_hist_mm": {"bin_edges": POS_BINS_MM.tolist(),
                        "counts": np.histogram(np.clip(pos_mm, POS_BINS_MM[0], POS_BINS_MM[-1]), POS_BINS_MM)[0].tolist()},
        "rot_hist_deg": {"bin_edges": ROT_BINS_DEG.tolist(),
                         "counts": np.histogram(np.clip(rot_deg, ROT_BINS_DEG[0], ROT_BINS_DEG[-1]), ROT_BINS_DEG)[0].tolist()},
        "frac_pos_err_above_mm": {str(c): float((pos_mm > c).mean()) for c in POS_CANDIDATES_MM},
        "frac_rot_err_above_deg": {str(c): float((rot_deg > c).mean()) for c in ROT_CANDIDATES_DEG},
        "frac_converged": float(conv.mean()),
        "n_reach_failures": int(reach_failure(res.pos_err_m[side], res.rot_err_rad[side]).sum()),
        "frac_reach_failures": float(reach_failure(res.pos_err_m[side], res.rot_err_rad[side])[v].mean()) if v.any() else None,
        "segments_pos_err_above_10mm": segments(np.nan_to_num(res.pos_err_m[side], nan=0) > 0.01, frame_index),
        "target_dist_from_shoulder_m": {"converged": pct(reach[conv]), "not_converged": pct(reach[~conv])},
        "arm_at_limit_frac": dict(zip(names, res.at_arm_limit[side][v].mean(0).tolist())),
        "start_used_counts": {k: int((res.start_used[side][v] == i).sum()) if i < 2 else int((res.start_used[side][v] >= 2).sum())
                              for i, k in enumerate(("warm", "home", "random_seed"))},
        "n_iters": pct(res.n_iters[side][v]),
    }


def free_waist_metrics(g: g1_model.G1Wuji, res, diag: dict) -> dict:
    """AS-B08 diagnostic summary: reach failures with the waist free vs the fixed-waist result."""
    out = {}
    for s in jo.SIDES:
        base = reach_failure(res.pos_err_m[s], res.rot_err_rad[s])
        free = reach_failure(diag[s]["pos_err_m"], diag[s]["rot_err_rad"])
        out[s] = {"n_reach_failures_fixed_waist": int(base.sum()), "n_reach_failures_free_waist": int(free.sum()),
                  "n_recovered": int((base & ~free).sum()), "n_new_failures": int((~base & free).sum()),
                  "pos_err_mm_free_waist": {k: (None if x is None else x * 1e3) for k, x in pct(diag[s]["pos_err_m"]).items()}}
    w = np.degrees(diag["q_waist"])
    leaning = np.isfinite(w).all(1) & (np.abs(w).max(1) > 1.0)
    out["waist"] = {"joints": list(g1_model.WAIST_JOINTS), "n_frames_waist_used_gt_1deg": int(leaning.sum()),
                    "abs_deg_p95_when_used": None, "frac_at_limit_when_used": None}
    if leaning.any():
        ids = [mujoco.mj_name2id(g.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in g1_model.WAIST_JOINTS]
        lo, hi = g.model.jnt_range[ids].T
        q = np.radians(w[leaning])
        out["waist"]["abs_deg_p95_when_used"] = np.percentile(np.abs(w[leaning]), 95, 0).tolist()
        out["waist"]["frac_at_limit_when_used"] = ((q <= lo + 0.02) | (q >= hi - 0.02)).mean(0).tolist()
    return out


def hist_png(metrics: dict, path) -> None:
    """2 x 2 small multiples (rows: position / rotation error; cols: left / right), log-x, one series each."""
    pw, ph, ml, mb, mt = 520, 260, 60, 50, 40
    surface, ink, ink2, grid, bar = (251, 252, 252), (30, 30, 30), (110, 110, 110), (225, 225, 225), (214, 120, 42)
    img = np.full((2 * ph + 40, 2 * pw, 3), surface, np.uint8)  # BGR
    cv2.putText(img, "B3 arm IK error per frame (log x). Dashed: candidate reach-failure threshold (W1-17)", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, ink, 1, cv2.LINE_AA)
    for r, (key, unit, thr) in enumerate((("pos_hist_mm", "palm position error (mm)", 10.0),
                                          ("rot_hist_deg", "palm rotation error (deg)", 5.0))):
        for c, side in enumerate(jo.SIDES):
            h = metrics["sides"][side][key]
            edges, counts = np.log10(h["bin_edges"]), np.array(h["counts"])
            x0, y0 = c * pw + ml, 40 + r * ph + mt
            w, hh = pw - ml - 20, ph - mt - mb
            xs = lambda e: int(x0 + (e - edges[0]) / (edges[-1] - edges[0]) * w)  # noqa: E731
            top = max(1, counts.max())
            for k in range(0, 5):
                y = int(y0 + hh - k / 4 * hh)
                cv2.line(img, (x0, y), (x0 + w, y), grid, 1)
                cv2.putText(img, f"{int(round(top * k / 4))}", (x0 - 40, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, ink2, 1, cv2.LINE_AA)
            for i, n in enumerate(counts):
                if n:
                    a, b = xs(edges[i]) + 1, xs(edges[i + 1]) - 1
                    cv2.rectangle(img, (a, int(y0 + hh - n / top * hh)), (b, y0 + hh), bar, -1)
            for e in range(int(np.ceil(edges[0])), int(np.floor(edges[-1])) + 1):
                cv2.putText(img, f"1e{e}", (xs(e) - 12, y0 + hh + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, ink2, 1, cv2.LINE_AA)
            xt = xs(np.log10(thr))
            for y in range(y0, y0 + hh, 8):
                cv2.line(img, (xt, y), (xt, y + 4), ink, 1)
            cv2.putText(img, f"{thr:g}", (xt + 3, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ink, 1, cv2.LINE_AA)
            n = sum(counts)
            cv2.putText(img, f"{side}: {unit}, n={n} frames", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ink, 1, cv2.LINE_AA)
            cv2.putText(img, "frames", (c * pw + 6, y0 + hh // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, ink2, 1, cv2.LINE_AA)
    imageio.imwrite(path, img[..., ::-1])


def stills_png(session: str, rt: dict, res, sample: list[int], path, placed: bool = True) -> None:
    g = placed_g1(W, H, placed)
    move_camera_to_reference(g, g1_model.HEAD_CAMERA)  # the human's viewpoint, where the targets come from
    renderer = mujoco.Renderer(g.model, H, W)
    head = mujoco.mj_name2id(g.model, mujoco.mjtObj.mjOBJ_CAMERA, g1_model.HEAD_CAMERA)
    behind = g1_behind_camera(g)
    rows = []
    for t in sample:
        g.data.qpos[:] = g.q_home
        pts = []
        for s in jo.SIDES:  # playback arrays: held sides show the held pose, without a target (B4)
            g.data.qpos[g.arm_qpos_adr[s]] = rt[f"{s}_q_g1_arm_cmd"][t]
            g.hands[s].set_q_sdk(g.data, rt[f"{s}_q_wuji_cmd"][t])
            if not res.held[s][t]:
                pts.append((res.T_world_from_palm_target[s][t][:3, 3], s))
        mujoco.mj_kinematics(g.model, g.data)
        mujoco.mj_camlight(g.model, g.data)
        tiles = []
        for cam in (head, behind):
            renderer.update_scene(g.data, cam)
            for p, s in pts:
                add_spheres(renderer.scene, [p], TARGET_RGBA, 0.012)
                add_spheres(renderer.scene, [g.data.xpos[g.palm_body_ids[s]]], REACHED_RGBA, 0.008)
            tiles.append(renderer.render().copy())
        row = np.hstack(tiles)
        msg = f"{session} frame {int(rt['frame_index'][t])}: " + "  ".join(
            f"{s} held ({'out of view' if not rt[f'{s}_visible'][t] else 'not retargeted'})" if res.held[s][t] else
            f"{s} pos {res.pos_err_m[s][t] * 1e3:.1f} mm rot {np.degrees(res.rot_err_rad[s][t]):.1f} deg" for s in jo.SIDES)
        cv2.putText(row, msg, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(row, "original head camera = human view (AS-01)", (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(row, "from behind (robot left = left); green = target, red = palm", (W + 8, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        rows.append(row)
    renderer.close()
    imageio.imwrite(path, np.vstack(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="session_032")
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--placement", choices=("b5", "none"), default="b5",
                    help="b5: W1-A06 pelvis offset + bent legs (default); none: pre-B5 home pose")
    args = ap.parse_args()
    placed = args.placement == "b5"

    rt_path = REPO_ROOT / f"data/derived/retarget/{args.session}.npz"
    rt, meta = derived_io.read_npz(rt_path)
    derived_io.check_retarget(rt)
    rt_sha_in = wm.sha256(rt_path)

    ik = ArmIK(placed_g1(placed=placed), T_world_from_cam=reference_T_world_from_cam())  # targets fixed (W1-B16)
    res = ik.run({s: rt[f"{s}_T_cam_from_palm"] for s in jo.SIDES}, {s: rt[f"{s}_valid"] for s in jo.SIDES})
    diag = ik.run_free_waist({s: rt[f"{s}_T_cam_from_palm"] for s in jo.SIDES}, {s: rt[f"{s}_valid"] for s in jo.SIDES})
    rt["q_waist_free_waist_diag"] = diag["q_waist"]
    for s in jo.SIDES:
        rt[f"{s}_ik_pos_err_m_free_waist_diag"] = diag[s]["pos_err_m"]
        rt[f"{s}_ik_rot_err_rad_free_waist_diag"] = diag[s]["rot_err_rad"]
        rt[f"{s}_reach_failure"] = reach_failure(res.pos_err_m[s], res.rot_err_rad[s])
        rt[f"g1_arm_joint_names_{s}"] = np.array(g1_model.arm_joint_names(s))
        home = ik.g.q_home[ik.g.arm_qpos_adr[s]]
        rt[f"{s}_q_g1_arm_cmd"], _, rt[f"{s}_arm_catching_up"] = hold.playback(
            res.q_arm[s], ~res.held[s], home, rt["timestamps_s"], hold.ARM_SPEED_CAP_RAD_S)
        rt |= {f"{s}_q_g1_arm": res.q_arm[s], f"{s}_ik_pos_err_m": res.pos_err_m[s], f"{s}_ik_rot_err_rad": res.rot_err_rad[s],
               f"{s}_T_world_from_palm_target": res.T_world_from_palm_target[s], f"{s}_ik_start_used": res.start_used[s],
               f"{s}_ik_n_iters": res.n_iters[s], f"{s}_arm_at_limit": res.at_arm_limit[s]}
    derived_io.check_retarget(rt)

    prov = provenance("scripts/b3_arm_ik.py")
    g1_xml = g1_model.g1_xml_path()
    sources = {str(rt_path) + " (input)": rt_sha_in, str(g1_xml): wm.sha256(g1_xml)} | {
        str(wm.mjcf_path(s)): wm.sha256(wm.mjcf_path(s)) for s in jo.SIDES}
    b3 = {**prov, "sources": sources, "params": res.params, "mount_pos": g1_model.MOUNT_POS,
          "mount_quat_wxyz": g1_model.MOUNT_QUAT_WXYZ, "assumptions": ["AS-01", "AS-02", "AS-09", "AS-12", "AS-B08"], "decisions": ["W1-B10", "W1-B11", "W1-B14", "W1-B15", "W1-B16", "W1-A06"],
          "placement": {"name": args.placement, "pelvis_offset_m": list(PELVIS_OFFSET_M) if placed else [0.0, 0.0, 0.0],
                        "leg_pitch_rad": LEG_PITCH_RAD if placed else None}}
    derived_io.write_npz(rt_path, rt, {k: v for k, v in meta.items() if k != "schema_version"} | {"b3": b3})

    metrics = {"session": args.session, "retarget": str(rt_path), "provenance": prov, "sources": sources,
               "params": res.params, "assumptions": b3["assumptions"], "placement": b3["placement"],
               "sides": {s: side_metrics(ik.g, s, res, rt["frame_index"]) for s in jo.SIDES}}
    for s in jo.SIDES:
        metrics["sides"][s]["hold_B4"] = hold_metrics(rt, res, s)
    metrics["free_waist_diagnostic_AS_B08"] = free_waist_metrics(ik.g, res, diag)
    out_dir = REPO_ROOT / "results" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    hist_png(metrics, out_dir / "ik_error_hist.png")

    worst = int(np.nanargmax(np.nan_to_num(res.pos_err_m["left"]) + np.nan_to_num(res.pos_err_m["right"])))
    T = len(rt["timestamps_s"])
    sample = sorted({0, T // 3, worst, 2 * T // 3, T - 1})
    stills_png(args.session, rt, res, sample, out_dir / "g1_stills.png", placed)
    for s in jo.SIDES:
        m = metrics["sides"][s]
        print(s, json.dumps({k: m[k] for k in ("pos_err_mm", "rot_err_deg", "frac_converged", "segments_pos_err_above_10mm")}))
    print(f"wrote {rt_path} and {out_dir}")


if __name__ == "__main__":
    main()
