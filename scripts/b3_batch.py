"""DoD 4: run B2 (retarget) and B3 (arm IK) on every unscrew_cap session and summarize per session.

Sessions: the unscrew_cap rows of data/raw/ego-clinical-task/session_index.csv (A3's authoritative list).
Per session it runs scripts/b2_retarget.py and scripts/b3_arm_ik.py (each in its own process, same
commands as for session_032) with exp-ids <exp_id>/<session>/{b2,b3}, then writes
  results/<exp_id>/summary.json        per session and side: Wuji at-limit % per joint, thumb bend-error check
                                       (AS-B06: flag thumb p95 > 25 deg), arm IK error mean/p95/max, reach
                                       failures (W1-B11), held frames and gap-edge steps (B4, W1-B14), arm
                                       at-limit % per joint, the free-waist diagnostic (AS-B08)
  results/<exp_id>/inventory_b.md      the markdown that --update-inventory puts into docs/DATA_INVENTORY.md
                                       (only the "Retargeting fields (workstream B)" section)

Usage: MUJOCO_GL=egl uv run python scripts/b3_batch.py --exp-id 20261001-b3-batch --update-inventory
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

from wuji_cap.common import derived_io
from wuji_cap.common import joint_order as jo
from wuji_cap.common.provenance import REPO_ROOT, provenance
from wuji_cap.sim import g1_model
from wuji_cap.sim.arm_ik import REACH_FAIL_POS_M, REACH_FAIL_ROT_RAD

THUMB_P95_FLAG_DEG = 25.0  # AS-B06 / W1-B09 resolution (2026-10-01)
INVENTORY = REPO_ROOT / "docs/DATA_INVENTORY.md"
SECTION = "## Retargeting fields (workstream B)"
JOINT_ABBR = {f: f[0].upper() for f in ("thumb", "index", "middle", "ring", "pinky")}


def sessions() -> list[str]:
    with open(REPO_ROOT / "data/raw/ego-clinical-task/session_index.csv") as f:
        return [r["session"] for r in csv.DictReader(f) if r["group"] == "unscrew_cap"]


def run(script: str, session: str, exp_id: str, extra: tuple[str, ...] = ()) -> None:
    cmd = [sys.executable, str(REPO_ROOT / script), "--session", session, "--exp-id", exp_id, *extra]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env={**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")})


def abbr(name: str) -> str:
    f, j = name.split("_joint")
    return JOINT_ABBR[f] + j


def summarize(session: str, exp_id: str) -> dict:
    b2 = json.loads((REPO_ROOT / f"results/{exp_id}/{session}/b2/metrics.json").read_text())
    b3 = json.loads((REPO_ROOT / f"results/{exp_id}/{session}/b3/metrics.json").read_text())
    rt, _ = derived_io.read_npz(REPO_ROOT / f"data/derived/retarget/{session}.npz")
    out = {"n_frames": int(len(rt["timestamps_s"])), "sides": {}}
    for s in jo.SIDES:
        m2, m3 = b2["sides"][s], b3["sides"][s]
        v = rt[f"{s}_valid"]
        thumb_p95 = {k: m2["bend_err_deg"][k]["err"]["p95"] for k in m2["bend_err_deg"] if k.startswith("thumb")}
        fw = b3["free_waist_diagnostic_AS_B08"][s]
        out["sides"][s] = {
            "n_valid": int(v.sum()),
            "wuji_at_limit_pct": {n: 100 * x for n, x in m2["at_limit_frac"].items()},
            "wuji_at_limit_any_pct": None if m2["at_limit_any_frac"] is None else 100 * m2["at_limit_any_frac"],
            "thumb_bend_err_p95_deg": thumb_p95,
            "thumb_flag": any(x is not None and x > THUMB_P95_FLAG_DEG for x in thumb_p95.values()),
            "finger_bend_err_p50_deg_range": [min(m2["bend_err_deg"][k]["err"]["p50"] for k in m2["bend_err_deg"]
                                                  if not k.startswith("thumb") and not k.endswith("knuckle")),
                                              max(m2["bend_err_deg"][k]["err"]["p50"] for k in m2["bend_err_deg"]
                                                  if not k.startswith("thumb") and not k.endswith("knuckle"))],
            "ik_pos_err_mm": m3["pos_err_mm"], "ik_rot_err_deg": m3["rot_err_deg"],
            "n_reach_failures": m3["n_reach_failures"], "frac_reach_failures": m3["frac_reach_failures"],
            "n_held": m2["hold_B4"]["n_held"], "n_not_visible": m2["hold_B4"]["n_not_visible"],
            "frac_held": m2["hold_B4"]["frac_held"], "longest_gap_s": m2["hold_B4"]["longest_gap_s"],
            "gaps_frame_index": m2["hold_B4"]["gaps_frame_index"],
            "arm_cmd_playback": m3["hold_B4"]["q_g1_arm_cmd_playback"], "wuji_cmd_playback": m2["hold_B4"]["q_wuji_cmd_playback"],
            "reach_failure_segments": m3["segments_pos_err_above_10mm"],
            "arm_at_limit_pct": {n: 100 * x for n, x in m3["arm_at_limit_frac"].items()},
            "free_waist": {k: fw[k] for k in ("n_reach_failures_free_waist", "n_recovered", "n_new_failures")},
        }
    out["waist_free_waist_diag"] = b3["free_waist_diagnostic_AS_B08"]["waist"]
    return out


def fmt(x, nd=1) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def inventory_md(summary: dict, exp_id: str) -> str:
    S = summary["sessions"]
    lines = [SECTION, "",
             f"B workstream, 2026-10-01. Evidence: [batch summary](../results/{exp_id}/summary.json); per session "
             f"`results/{exp_id}/<session>/{{b2,b3}}/metrics.json`, `ik_error_hist.png`, `g1_stills.png`. "
             "Kinematic only (AS-12); IK depends on AS-01 (camera mount), AS-02 (wrist mount), AS-09 (fixed pelvis, legs, waist; robot placed per W1-A06 when summary.json says placement b5). "
             f"Reach failure (W1-B11): palm position error > {REACH_FAIL_POS_M * 1e3:g} mm or rotation error > "
             f"{np.degrees(REACH_FAIL_ROT_RAD):g}°. At limit: within 0.02 rad of a joint limit. Statistics are over usable frames only "
             "(B4, W1-B14): A's keypoints valid and the 5 knuckles inside the image (W1-B15; the wrist may be past the edge), an "
             "image-bounds test, not an occlusion test. On the other frames (\"held\") the contract fields are NaN and the playback "
             "arrays `*_cmd` hold the last pose (home before the first usable frame); after a gap they return to the tracked pose "
             "at no more than 2 rad/s (arm) and 3 rad/s (fingers).",
             "",
             "| Session | Side | Wuji any joint at limit | Arm IK position error mean / p95 / max (mm) | Arm IK rotation error mean / p95 / max (°) "
             "| Held frames (% of frames; longest gap) | Reach failures (% of usable frames) | Reach failures with free waist (AS-B08 diagnostic) | Thumb bend error p95 (°, CMC / MCP / IP) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for sess, d in S.items():
        for s in jo.SIDES:
            x = d["sides"][s]
            p, r = x["ik_pos_err_mm"], x["ik_rot_err_deg"]
            th = x["thumb_bend_err_p95_deg"]
            lines.append(
                f"| {sess} | {s} | {fmt(x['wuji_at_limit_any_pct'])} % | {fmt(p['mean'])} / {fmt(p['p95'])} / {fmt(p['p100'])} "
                f"| {fmt(r['mean'])} / {fmt(r['p95'])} / {fmt(r['p100'])} | {x['n_held']} ({fmt(100 * x['frac_held'])} %; "
                f"{fmt(x['longest_gap_s'])} s) | {x['n_reach_failures']} ({fmt(100 * x['frac_reach_failures'])} %) "
                f"| {x['free_waist']['n_reach_failures_free_waist']} "
                f"| {' / '.join(fmt(th[k]) for k in ('thumb_knuckle', 'thumb_middle', 'thumb_distal'))}"
                f"{' ⚑' if x['thumb_flag'] else ''} |")
    lines += ["", "Percent of usable frames at a joint limit, per Wuji joint (T/I/M/R/P = thumb/index/middle/ring/pinky, "
              "1-4 = SDK joint number; columns where every row is 0 are left out):", ""]
    names = [n for n in jo.SDK_JOINT_NAMES
             if any(d["sides"][s]["wuji_at_limit_pct"][n] > 0 for d in S.values() for s in jo.SIDES)]
    lines.append("| Session | Side | " + " | ".join(abbr(n) for n in names) + " |")
    lines.append("|---|---|" + "---:|" * len(names))
    for sess, d in S.items():
        for s in jo.SIDES:
            lines.append(f"| {sess} | {s} | " + " | ".join(fmt(d["sides"][s]["wuji_at_limit_pct"][n]) for n in names) + " |")
    lines.append(f"\nAll other Wuji joints: 0 % in every session ({', '.join(abbr(n) for n in jo.SDK_JOINT_NAMES if n not in names)}).")
    lines += ["", "Percent of usable frames at a joint limit, per G1 arm joint:", "",
              "| Session | Side | " + " | ".join(g1_model.G1_ARM_JOINTS) + " |", "|---|---|" + "---:|" * 7]
    for sess, d in S.items():
        for s in jo.SIDES:
            a = d["sides"][s]["arm_at_limit_pct"]
            lines.append(f"| {sess} | {s} | " + " | ".join(fmt(a[n]) for n in g1_model.arm_joint_names(s)) + " |")
    lines += ["", "Reading the table:",
              "- Reach failures come in contiguous segments (`reach_failure_segments` in the summary). Targets the G1 arm cannot "
              "reach lie further than about 0.39 m from the shoulder (AS-B08).",
              "- Held frames: the hand is out of view (or was not retargeted), so the robot keeps its last usable pose. Every "
              "session starts with both hands out of view; the robot holds the home pose until each hand appears. "
              "Playback speeds and catch-up durations are in the summary (`arm_cmd_playback`, `wuji_cmd_playback`).",
              "- The free-waist column is a diagnostic (targets fixed in the world, waist free, head camera not moved), "
              "not the robot motion in the retarget files.",
              f"- ⚑ = thumb bend-error p95 above {THUMB_P95_FLAG_DEG:g}° (AS-B06 check). The bend error compares Wuji with A's "
              "keypoints, not with the true hand.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--update-inventory", action="store_true")
    ap.add_argument("--placement", choices=("b5", "none"), default="b5", help="passed to b3_arm_ik.py")
    args = ap.parse_args()

    names = args.sessions or sessions()
    if not args.summarize_only:
        for sess in names:
            run("scripts/b2_retarget.py", sess, f"{args.exp_id}/{sess}/b2")
            run("scripts/b3_arm_ik.py", sess, f"{args.exp_id}/{sess}/b3", ("--placement", args.placement))
    summary = {"exp_id": args.exp_id, "sessions": {s: summarize(s, args.exp_id) for s in names},
               "thumb_p95_flag_deg": THUMB_P95_FLAG_DEG, "reach_fail_pos_m": REACH_FAIL_POS_M,
               "reach_fail_rot_deg": float(np.degrees(REACH_FAIL_ROT_RAD)),
               "assumptions": ["AS-01", "AS-02", "AS-09", "AS-12", "AS-B06", "AS-B08"], "placement": args.placement,
               "provenance": provenance("scripts/b3_batch.py")}
    out_dir = REPO_ROOT / "results" / args.exp_id
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    md = inventory_md(summary, args.exp_id)
    (out_dir / "inventory_b.md").write_text(md)
    if args.update_inventory:
        text = INVENTORY.read_text()
        i = text.index(SECTION)
        j = text.index("\n## ", i + len(SECTION))
        INVENTORY.write_text(text[:i] + md + text[j:])
        print(f"updated {INVENTORY}")
    print(md)


if __name__ == "__main__":
    main()
