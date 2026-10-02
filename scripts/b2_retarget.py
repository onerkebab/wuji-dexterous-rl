"""B2: retarget one session's keypoints to Wuji joint angles; write the contract-v2 retarget npz + metrics.

Reads  data/derived/keypoints/<session>.npz   (A2 output, or the B2 stub)
Writes data/derived/retarget/<session>.npz     (G1 arm fields are NaN until B3)
       results/<exp_id>/metrics.json           (per side: scale, palm fit, tip error, at-limit, AS-03 offset,
                                                 human-vs-Wuji bend-angle error; for the stub also the error
                                                 against the true joint angles; B4 held frames and gaps)
B4 (W1-B14, W1-B15): frames with a knuckle outside the image (session intrinsics from the raw metadata) are not
retargeted; statistics cover usable frames only. Without metadata (stub) every frame counts as in view.
--target-mode scaled reproduces the superseded W1-B05 targets (comparison only; W1-B08).

Usage: uv run python scripts/b2_retarget.py --session stub_session_032 --exp-id 20260930-b2-stub
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.spatial.transform import Rotation

from wuji_cap.common import derived_io, frames
from wuji_cap.common import joint_order as jo
from wuji_cap.common.provenance import REPO_ROOT, provenance
from wuji_cap.sim import hand_keypoints as hk
from wuji_cap.sim import hold
from wuji_cap.sim import wuji_model as wm
from wuji_cap.sim.render import load_camera
from wuji_cap.sim.retarget import SideResult, retarget_session

FINGER_NAMES = list(jo.FINGERS)


def pct(x: np.ndarray, qs=(50, 95, 100)) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{q}": None for q in qs}
    return {f"p{q}": float(np.percentile(x, q)) for q in qs}


def palm_offset_stats(T_cam_from_wrist: np.ndarray, res: SideResult) -> dict:
    """AS-03: T_wrist_from_palm per frame (wrist = keypoint file's wrist frame, palm = fitted Wuji wrist)."""
    ok = res.valid & np.isfinite(T_cam_from_wrist).all(axis=(1, 2))
    if not ok.any():
        return {"n_frames": 0}
    T = frames.inv_T(T_cam_from_wrist[ok]) @ res.T_cam_from_palm[ok]
    rots = Rotation.from_matrix(T[:, :3, :3])
    mean = rots.mean()
    dev = (mean.inv() * rots).magnitude()
    t = T[:, :3, 3]
    return {
        "n_frames": int(ok.sum()),
        "rotvec_mean_rad": mean.as_rotvec().tolist(),
        "rot_angle_mean_deg": float(np.degrees(mean.magnitude())),
        "rot_dev_from_mean_deg": {k: float(np.degrees(v)) for k, v in pct(dev).items()},
        "translation_median_m": np.median(t, 0).tolist(),
        "translation_std_m": t.std(0).tolist(),
    }


def bend_error(side: str, kp: dict, res: SideResult) -> dict:
    """Independent quality check: bend angles of the human keypoints vs the retargeted Wuji FK pose,
    per finger and joint (knuckle, middle, distal). Knuckle bends include the palm-shape difference
    (the human and Wuji wrist->knuckle geometry differ), so read the middle/distal ones first."""
    v = res.valid
    if not v.any():
        return {}
    kin = hk.WujiKinematics(side)
    human = hk.bend_angles(kp[f"{side}_kp21_cam"][v].astype(float))
    wuji = hk.bend_angles(np.array([kin.keypoints21(q) for q in res.q_wuji[v]]))
    err = np.degrees(np.abs(wuji - human))
    out = {}
    for i, f in enumerate(FINGER_NAMES):
        for j, jn in enumerate(("knuckle", "middle", "distal")):
            out[f"{f}_{jn}"] = {"err": pct(err[:, i, j]), "human_p50": float(np.degrees(np.median(human[:, i, j])))}
    return out


def hold_metrics(out: dict, side: str) -> dict:
    """B4: held frames (not usable), gaps, and the finger playback speed (catch-up after gaps)."""
    held, ts = out[f"{side}_held"], out["timestamps_s"]
    gaps = hold.gap_segments(held)
    longest = max(gaps, key=lambda g: g[1] - g[0]) if gaps else None
    return {"n_held": int(held.sum()), "frac_held": float(held.mean()),
            "n_not_visible": int((~out[f"{side}_visible"]).sum()),
            "gaps_frame_index": [[int(out["frame_index"][a]), int(out["frame_index"][b - 1])] for a, b in gaps],
            "longest_gap_frames": None if longest is None else longest[1] - longest[0],
            "longest_gap_s": None if longest is None else float(ts[min(longest[1], len(ts) - 1)] - ts[longest[0]]),
            "speed_cap_rad_s": hold.WUJI_SPEED_CAP_RAD_S,
            "q_wuji_cmd_playback": hold.playback_stats(out[f"{side}_q_wuji_cmd"], held, out[f"{side}_wuji_catching_up"], ts)}


def side_metrics(side: str, kp: dict, res: SideResult, out: dict) -> dict:
    v = res.valid
    m = {
        "n_frames": int(len(v)),
        "n_valid_in": int(kp[f"{side}_valid"].sum()),
        "n_visible_in": int((kp[f"{side}_valid"] & out[f"{side}_visible"]).sum()),
        "n_valid_out": int(v.sum()),
        "n_opt_fail": res.n_opt_fail,
        "n_palm_fit_fail": int((np.isfinite(res.palm_rms_m) & ~v).sum()) - res.n_opt_fail,
        "scale_wuji_over_human": res.scale,
        "palm_rms_mm": {k: (None if x is None else x * 1e3) for k, x in pct(res.palm_rms_m).items()},
        "tip_err_mm": {f: {k: (None if x is None else x * 1e3) for k, x in pct(res.tip_err_m[:, i]).items()}
                       for i, f in enumerate(FINGER_NAMES)},
        "at_limit_frac": dict(zip(jo.SDK_JOINT_NAMES, (res.at_limit[v].mean(0) if v.any() else np.zeros(20)).tolist())),
        "at_limit_any_frac": float(res.at_limit[v].any(1).mean()) if v.any() else None,
        "palm_offset_AS03": palm_offset_stats(kp[f"{side}_T_cam_from_wrist"], res),
        "bend_err_deg": bend_error(side, kp, res),
        "hold_B4": hold_metrics(out, side),
    }
    q_true = kp.get(f"stub_{side}_q_true_sdk")
    if q_true is not None:
        e = np.abs(res.q_wuji[v] - q_true[v])
        m["stub_joint_err_rad"] = {"all": pct(e), "per_joint_p95": dict(zip(jo.SDK_JOINT_NAMES, np.percentile(e, 95, 0).tolist()))}
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="stub_session_032")
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--target-mode", default="direction", choices=("direction", "scaled"))
    ap.add_argument("--no-npz", action="store_true", help="metrics only; leave data/derived/retarget untouched")
    args = ap.parse_args()

    kp_path = REPO_ROOT / f"data/derived/keypoints/{args.session}.npz"
    kp, kp_meta = derived_io.read_npz(kp_path)
    derived_io.check_keypoints(kp)

    cam = load_camera(args.session)
    out, results = retarget_session(kp, args.target_mode, cam)
    metrics = {"session": args.session, "target_mode": args.target_mode, "stub": bool(kp_meta.get("params", {}).get("stub", False)),
               "visibility": "5 knuckles inside the image, session intrinsics (W1-B15)" if cam is not None
               else "no session metadata: every frame counts as in view", "sides": {}}
    for side, res in results.items():
        metrics["sides"][side] = side_metrics(side, kp, res, out)
        print(side, json.dumps({k: metrics["sides"][side][k] for k in ("n_valid_in", "n_visible_in", "n_valid_out", "scale_wuji_over_human")}))
    params = {side: res.params for side, res in results.items()}
    derived_io.check_retarget(out)

    prov = provenance("scripts/b2_retarget.py")
    sources = {str(kp_path): wm.sha256(kp_path)} | {str(wm.urdf_path(s)): wm.sha256(wm.urdf_path(s)) for s in jo.SIDES}
    out_path = REPO_ROOT / f"data/derived/retarget/{args.session}.npz"
    if args.no_npz:
        out_path = None
    else:
        derived_io.write_npz(out_path, out, {**prov, "sources": sources, "params": params,
                                             "wuji_description_commit": wm.wuji_description_commit()})
    res_dir = REPO_ROOT / "results" / args.exp_id
    res_dir.mkdir(parents=True, exist_ok=True)
    metrics |= {"provenance": prov, "keypoints": str(kp_path), "retarget": None if out_path is None else str(out_path),
                "assumptions": ["AS-B03", "AS-B04", "AS-B05", "AS-B06", "AS-B07", "AS-B01", "AS-B02", "AS-12"], "decisions": ["W1-B14", "W1-B15"]}
    (res_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out_path or '(no npz)'} and {res_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
