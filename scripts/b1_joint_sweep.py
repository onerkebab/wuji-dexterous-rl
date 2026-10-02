"""B1: Wuji Hand 2 joint sweep in SDK order, both hands, with SDK and MJCF names on screen.

Sweeps SDK joint 0..19 one at a time (0 -> upper limit -> lower limit -> 0), others held at 0.
The bodies moved by the active joint are highlighted. Also writes a contact sheet (each joint at
its upper limit) and metrics.json with the joint table and the MuJoCo-vs-Pinocchio fingertip check.

Usage: MUJOCO_GL=egl uv run python scripts/b1_joint_sweep.py [--exp-id 20260929-b1-joint-sweep]
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
import pinocchio as pin

from wuji_cap.common import joint_order as jo
from wuji_cap.common.provenance import REPO_ROOT, provenance
from wuji_cap.sim import wuji_model as wm

W, H = 640, 540  # per-hand panel
BANNER = 110
FPS = 30
SWEEP_S = 2.0
HOLD_S = 1.0
CAM_OFFAXIS_DEG = 55.0
HIGHLIGHT = np.array([1.0, 0.45, 0.1, 1.0])


def build_render_model(side: jo.Side) -> mujoco.MjModel:
    """Beta 2 MJCF with the hand turned fingers-up, plus lights and a plain backdrop (visual only)."""
    spec = mujoco.MjSpec.from_file(str(wm.mjcf_path(side)))
    spec.body(jo.side_prefix(side) + "wrist").quat = [0.0, 0.0, 1.0, 0.0]  # 180 deg about y: fingers +z
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    spec.worldbody.add_light(pos=[0.3, -0.5, 0.6], dir=[-0.3, 0.5, -0.4], diffuse=[0.6, 0.6, 0.6])
    spec.worldbody.add_light(pos=[-0.3, 0.5, 0.6], dir=[0.3, -0.5, -0.4], diffuse=[0.4, 0.4, 0.4])
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.93, 0.94, 0.96],
                     rgb2=[0.75, 0.78, 0.82], width=64, height=64)
    return spec.compile()


def descendants(model: mujoco.MjModel, root: int) -> set[int]:
    out = {root}
    for b in range(model.nbody):  # bodies are topologically ordered
        if model.body_parentid[b] in out:
            out.add(b)
    return out


class Panel:
    def __init__(self, side: jo.Side):
        self.side = side
        self.model = build_render_model(side)
        self.hand = wm.load_hand(side, self.model)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.0, 0.0, 0.11]
        self.cam.distance = 0.30
        # Palm faces -y (left) / +y (right) after the flip; view the palm, 55 deg off-axis so flexion is visible.
        self.cam.azimuth = 90.0 + CAM_OFFAXIS_DEG if side == "left" else -90.0 - CAM_OFFAXIS_DEG
        self.cam.elevation = -15.0
        self.rgba0 = self.model.geom_rgba.copy()
        self.matid0 = self.model.geom_matid.copy()
        self.joint_geoms = []
        for j in self.hand.jnt_ids:
            bodies = descendants(self.model, self.model.jnt_bodyid[j])
            self.joint_geoms.append(np.flatnonzero(np.isin(self.model.geom_bodyid, list(bodies))))

    def render(self, q_sdk: np.ndarray, active: int | None) -> np.ndarray:
        self.model.geom_rgba[:] = self.rgba0
        self.model.geom_matid[:] = self.matid0
        if active is not None:
            g = self.joint_geoms[active]
            self.model.geom_matid[g] = -1
            self.model.geom_rgba[g] = HIGHLIGHT
        self.hand.set_q_sdk(self.data, q_sdk)
        mujoco.mj_kinematics(self.model, self.data)
        self.renderer.update_scene(self.data, self.cam)
        return self.renderer.render().copy()


def sweep_profile(lo: float, hi: float, n: int) -> np.ndarray:
    """0 -> hi -> 0 -> lo -> 0 over n samples (sinusoid, scaled separately on each side)."""
    s = np.sin(np.linspace(0.0, 2.0 * np.pi, n))
    return np.where(s >= 0, s * hi, -s * lo)


def put(img, text, org, scale=0.6, color=(20, 20, 20), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def compose(panels: list[Panel], frames: list[np.ndarray], i: int | None, q: float, rng: np.ndarray) -> np.ndarray:
    top = np.full((BANNER, W * len(panels), 3), 255, np.uint8)
    if i is None:
        put(top, "Wuji Hand 2 (Beta 2 MJCF) - zero pose, SDK joint sweep follows", (15, 35), 0.8, thick=2)
    else:
        put(top, f"SDK {i:2d}: {jo.SDK_JOINT_NAMES[i]}", (15, 38), 1.1, (10, 60, 160), 2)
        lo, hi = np.degrees(rng)
        put(top, f"q = {q:+.3f} rad ({np.degrees(q):+6.1f} deg)   range [{lo:+.0f}, {hi:+.0f}] deg",
            (15, 72), 0.7)
    put(top, "kinematic only (AS-12); orange = bodies moved by the active joint", (15, 100), 0.5, (90, 90, 90))
    tiles = []
    for p, f in zip(panels, frames):
        f = f.copy()
        put(f, f"{p.side} hand", (12, 26), 0.7, thick=2)
        if i is not None:
            put(f, f"MJCF joint: {jo.model_joint_names(p.side)[i]}", (12, H - 38), 0.6)
            put(f, f"actuator:   {jo.mjcf_actuator_names(p.side)[i]}", (12, H - 14), 0.6)
        tiles.append(f)
    return np.vstack([top, np.hstack(tiles)])


def cross_fk_check(side: jo.Side, n: int = 1000, seed: int = 0) -> dict:
    hand = wm.load_hand(side)
    d = mujoco.MjData(hand.model)
    pm = pin.buildModelFromUrdf(str(wm.urdf_path(side)))
    pd = pm.createData()
    names = [nm for k, nm in enumerate(pm.names) if pm.nqs[k] > 0]
    wf = pm.getFrameId(jo.side_prefix(side) + "wrist")
    tf = [pm.getFrameId(t) for t in jo.tip_names(side)]

    def pin_tips(q):
        pin.framesForwardKinematics(pm, pd, q)
        Wi = pd.oMf[wf].inverse()
        return np.array([(Wi * pd.oMf[f]).translation for f in tf])

    rng = np.random.default_rng(seed)
    lo, hi = hand.joint_range.T
    err, err_unmapped = [], []
    for _ in range(n):
        q = rng.uniform(lo, hi)
        hand.set_q_sdk(d, q)
        mujoco.mj_kinematics(hand.model, d)
        t = hand.tips_in_wrist(d)
        err.append(np.linalg.norm(pin_tips(jo.from_sdk(q, names, side)) - t, axis=1).max())
        err_unmapped.append(np.linalg.norm(pin_tips(q) - t, axis=1).max())
    return {
        "n_samples": n,
        "seed": seed,
        "pinocchio_dof_order": names,
        "sdk_to_pinocchio_perm": jo.permutation(jo.model_joint_names(side), names).tolist(),
        "max_tip_err_mm": 1e3 * float(np.max(err)),
        "unmapped_negative_control_min_of_max_tip_err_mm": 1e3 * float(np.min(err_unmapped)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp-id", default="20260929-b1-joint-sweep")
    args = ap.parse_args()
    out = REPO_ROOT / "results" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)

    panels = [Panel("left"), Panel("right")]
    rngs = panels[0].hand.joint_range
    assert np.array_equal(rngs, panels[1].hand.joint_range)
    n_sweep, n_hold = int(SWEEP_S * FPS), int(HOLD_S * FPS)

    video = out / "wuji_joint_sweep.mp4"
    sheet_tiles = []
    n_frames = 0
    with imageio.get_writer(video, fps=FPS, codec="libx264", quality=8, macro_block_size=1) as w:
        zero = np.zeros(jo.N_JOINTS)
        f0 = compose(panels, [p.render(zero, None) for p in panels], None, 0.0, rngs[0])
        for _ in range(n_hold):
            w.append_data(f0)
            n_frames += 1
        for i in range(jo.N_JOINTS):
            for qi in sweep_profile(*rngs[i], n_sweep):
                q = zero.copy()
                q[i] = qi
                w.append_data(compose(panels, [p.render(q, i) for p in panels], i, qi, rngs[i]))
                n_frames += 1
            q = zero.copy()
            q[i] = rngs[i, 1]
            sheet_tiles.append(compose(panels, [p.render(q, i) for p in panels], i, q[i], rngs[i]))

    for p in panels:  # release EGL contexts before interpreter teardown
        p.renderer.close()

    rows = [np.hstack(sheet_tiles[r * 4:(r + 1) * 4]) for r in range(5)]
    sheet = cv2.resize(np.vstack(rows), None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    imageio.imwrite(out / "wuji_joint_sweep_upper_limits.png", sheet)

    joint_table = [
        {
            "sdk_index": i,
            "sdk_name": jo.SDK_JOINT_NAMES[i],
            "mjcf_joint": {s: jo.model_joint_names(s)[i] for s in jo.SIDES},
            "mjcf_actuator": {s: jo.mjcf_actuator_names(s)[i] for s in jo.SIDES},
            "range_deg": np.round(np.degrees(rngs[i]), 2).tolist(),
        }
        for i in range(jo.N_JOINTS)
    ]
    metrics = {
        "exp_id": args.exp_id,
        "task": "B1 (M0 task 3, joint-order module, MJCF only)",
        "label": "kinematic only (AS-12)",
        "provenance": provenance("scripts/b1_joint_sweep.py"),
        "wuji_description": {
            "commit": wm.wuji_description_commit(),
            **{f"{s}_mjcf_sha256": wm.sha256(wm.mjcf_path(s)) for s in jo.SIDES},
            **{f"{s}_urdf_sha256": wm.sha256(wm.urdf_path(s)) for s in jo.SIDES},
        },
        "video": {"path": video.name, "fps": FPS, "frames": n_frames, "sweep_s_per_joint": SWEEP_S},
        "contact_sheet": "wuji_joint_sweep_upper_limits.png",
        "joint_table": joint_table,
        "cross_fk_mujoco_vs_pinocchio": {s: cross_fk_check(s) for s in jo.SIDES},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    for s, c in metrics["cross_fk_mujoco_vs_pinocchio"].items():
        print(f"{s}: cross-FK max tip err {c['max_tip_err_mm']:.4f} mm "
              f"(unmapped control {c['unmapped_negative_control_min_of_max_tip_err_mm']:.1f} mm)")
    print(f"wrote {video} ({n_frames} frames), contact sheet, metrics.json in {out}")


if __name__ == "__main__":
    main()
