"""Human MediaPipe-21 keypoints (camera frame) -> 20 Wuji Beta 2 joint angles (SDK order).

Pipeline per frame (DECISIONS W1-B05, W1-B06):
1. Palm fit (``hand_keypoints.fit_palm``): rigid rotation + translation of the human keypoints onto
   the Wuji zero-pose palm points, at one Wuji/human scale per session and side (median of
   per-frame similarity fits). This replaces wuji-retargeting's MediaPipe frame estimate and
   ``mediapipe_rotation`` hand-tuning, and gives T_cam_from_palm for B3.
2. Direction targets (W1-B08, ``hand_keypoints.direction_targets``): human segment directions with
   Wuji segment lengths, chained from the Wuji knuckles. Supersedes W1-B05's "scaled keypoints".
   ``target_mode="scaled"`` keeps the W1-B05 behaviour for comparison only.
3. wuji-retargeting ``VectorOptimizer`` (@ 531f6ed, W1-04) on the Beta 2 URDF: 15 vectors
   wrist -> {PIP, DIP, tip} per finger (thumb: MCP, IP, tip), all scales 1.0. The wrist target is
   the Wuji wrist origin, so every target is a position in the Wuji wrist frame.
4. Pinocchio joint order -> SDK order through ``joint_order`` only.

Offline, so no low-pass filter (it adds lag); the optimizer's ``norm_delta`` term toward the
previous solution smooths instead. Invalid frames give NaN and reset the warm start, so nothing
is carried across a gap.

B4 (W1-B14, W1-B15): a frame is retargeted only if it is usable (A's valid AND the 5 knuckles in view,
``hold.knuckles_in_image``). ``{side}_q_wuji_cmd`` is the playback array (``hold.playback``: hold through gaps, home =
zero pose clipped to the limits, speed-capped return); ``{side}_q_wuji`` stays NaN on unusable frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from wuji_retargeting.opt import BaseOptimizer

from wuji_cap.common import joint_order as jo
from wuji_cap.common.joint_order import Side
from wuji_cap.sim import hand_keypoints as hk
from wuji_cap.sim import hold
from wuji_cap.sim import wuji_model as wm

AT_LIMIT_TOL_RAD = 0.02  # provisional (W1-17)
PALM_RMS_FAIL_M = 0.02  # palm fit worse than this -> frame marked invalid (provisional, W1-17)


def default_config(side: Side) -> dict:
    """VectorOptimizer config for the Beta 2 URDF. Key vectors use full URDF frame names."""
    names = hk.mp_frame_names(side)
    kvs = []
    for kp in range(1, 21):
        if kp in (1, 5, 9, 13, 17):  # knuckles do not move relative to the wrist on Wuji
            continue
        kvs.append({"origin": names[0], "task": names[kp], "origin_kp": 0, "task_kp": kp, "scale": 1.0})
    return {
        "optimizer": {
            "type": "VectorOptimizer",
            "hand_side": side,
            "urdf_path": str(wm.urdf_path(side)),
            # Required by BaseOptimizer even in vector mode; as in the package's Wuji Hand 2 example.
            "link_naming": {
                "prefix": jo.side_prefix(side),
                "palm": "wrist",
                "fingers": ["thumb", "index_finger", "middle_finger", "ring_finger", "pinky"],
                "pip": "{finger}_middle",
                "dip": "{finger}_distal",
                "tip": "{finger}_tip",
                "link1": "{finger}_proximal_abd",
            },
        },
        "retarget": {"huber_delta": 2.0, "norm_delta": 0.04, "key_vectors": kvs},
    }


@dataclass
class SideResult:
    valid: np.ndarray  # [T] bool
    q_wuji: np.ndarray  # [T, 20] SDK order
    at_limit: np.ndarray  # [T, 20] bool
    tip_err_m: np.ndarray  # [T, 5] Wuji FK tip vs target tip (W1-B08: direction-built target)
    T_cam_from_palm: np.ndarray  # [T, 4, 4]
    palm_rms_m: np.ndarray  # [T]
    scale: float
    n_opt_fail: int
    params: dict = field(default_factory=dict)


class WujiRetargeter:
    def __init__(self, side: Side, config: dict | None = None, target_mode: str = "direction"):
        if target_mode not in ("direction", "scaled"):
            raise ValueError(f"target_mode must be 'direction' or 'scaled', got {target_mode!r}")
        self.side = side
        self.target_mode = target_mode
        self.config = config if config is not None else default_config(side)
        self.optimizer = BaseOptimizer.from_config(self.config)
        self.kin = hk.WujiKinematics(side)
        names = tuple(self.optimizer.robot.dof_joint_names)
        if names != self.kin.pin_joint_names:
            raise ValueError("retargeter and FK disagree on the Pinocchio joint order")
        self.pin_joint_names = names

    def reset(self) -> None:
        self.optimizer.last_qpos = None

    def session_scale(self, kp21_cam: np.ndarray, valid: np.ndarray) -> float:
        s = [hk.palm_scale(kp21_cam[t], self.kin.palm_template) for t in np.flatnonzero(valid)]
        return float(np.median(s)) if s else float("nan")

    def targets(self, kp_palm: np.ndarray, q_ref: np.ndarray | None) -> np.ndarray:
        """Optimizer targets; ``q_ref`` (previous frame's solution, None after a reset) sets the Wuji
        segment lengths for direction targets."""
        if self.target_mode == "direction":
            ref = self.kin.zero_kp if q_ref is None else self.kin.keypoints21(q_ref)
            return hk.direction_targets(kp_palm, ref)
        return kp_palm

    def solve_frame(self, kp_palm: np.ndarray) -> tuple[np.ndarray, bool]:
        """One optimizer solve on Wuji-frame keypoints. Returns (q_sdk, ok)."""
        kp = kp_palm.copy()
        kp[0] = 0.0  # target vectors start at the Wuji wrist origin
        q_pin = np.asarray(self.optimizer.solve(kp), dtype=float)
        ok = self.optimizer.opt.last_optimize_result() > 0
        return jo.to_sdk(q_pin, self.pin_joint_names, self.side), ok

    def run(self, kp21_cam: np.ndarray, valid: np.ndarray, scale: float | None = None) -> SideResult:
        kp21_cam = np.asarray(kp21_cam, dtype=float)
        valid = np.asarray(valid, dtype=bool) & np.isfinite(kp21_cam).all(axis=(1, 2))
        T = len(kp21_cam)
        if scale is None:
            scale = self.session_scale(kp21_cam, valid)
        out = SideResult(
            valid=np.zeros(T, bool),
            q_wuji=np.full((T, jo.N_JOINTS), np.nan),
            at_limit=np.zeros((T, jo.N_JOINTS), bool),
            tip_err_m=np.full((T, 5), np.nan),
            T_cam_from_palm=np.full((T, 4, 4), np.nan),
            palm_rms_m=np.full(T, np.nan),
            scale=scale,
            n_opt_fail=0,
            params={"at_limit_tol_rad": AT_LIMIT_TOL_RAD, "palm_rms_fail_m": PALM_RMS_FAIL_M,
                    "palm_kp": list(hk.PALM_KP), "target_mode": self.target_mode, "config": self.config},
        )
        self.reset()
        q_prev = None
        for t in range(T):
            if not valid[t]:
                self.reset()
                q_prev = None
                continue
            fit = hk.fit_palm(kp21_cam[t], self.kin.palm_template, scale)
            out.T_cam_from_palm[t] = fit.T_cam_from_palm
            out.palm_rms_m[t] = fit.rms_m
            if fit.rms_m > PALM_RMS_FAIL_M:
                self.reset()
                q_prev = None
                continue
            target = self.targets(fit.kp_palm, q_prev)
            q, ok = self.solve_frame(target)
            if not ok:
                out.n_opt_fail += 1
                self.reset()
                q_prev = None
                continue
            q_prev = q
            out.valid[t] = True
            out.q_wuji[t] = q
            lo, hi = self.kin.q_lower_sdk, self.kin.q_upper_sdk
            out.at_limit[t] = (q <= lo + AT_LIMIT_TOL_RAD) | (q >= hi - AT_LIMIT_TOL_RAD)
            tips = self.kin.keypoints21(q)[list(hk.MP_TIPS)]
            out.tip_err_m[t] = np.linalg.norm(tips - target[list(hk.MP_TIPS)], axis=1)
        return out


G1_ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")


def wuji_home(kin: hk.WujiKinematics) -> np.ndarray:
    """Hold value before the first usable frame: the zero (open) pose, clipped to the joint limits."""
    return np.clip(np.zeros(jo.N_JOINTS), kin.q_lower_sdk, kin.q_upper_sdk)


def retarget_session(kp: dict, target_mode: str = "direction", cam: dict | None = None) -> tuple[dict, dict[str, SideResult]]:
    """Retarget both hands of a contract-v2 keypoint dict. Returns (retarget arrays, per-side results).

    ``cam``: session camera intrinsics for the palm-in-view test (B4); None = every frame in view (stub).
    G1 arm fields are NaN until B3. Keys beyond contract v2 (W1-B05): ``{side}_T_cam_from_palm``,
    ``{side}_palm_rms_m``, ``{side}_scale``; B4 (W1-B14, W1-B15): ``{side}_visible``, ``{side}_q_wuji_cmd``, ``{side}_held``,
    ``{side}_wuji_catching_up``.
    """
    T = len(kp["timestamps_s"])
    out = {
        "frame_index": kp["frame_index"],
        "timestamps_s": kp["timestamps_s"],
        "wuji_joint_names": np.array(jo.SDK_JOINT_NAMES),
    }
    results = {}
    for side in jo.SIDES:
        visible = hold.knuckles_in_image(kp[f"{side}_kp21_cam"], cam)
        rt = WujiRetargeter(side, target_mode=target_mode)
        res = rt.run(kp[f"{side}_kp21_cam"], kp[f"{side}_valid"] & visible)
        results[side] = res
        q_cmd, held, catching = hold.playback(res.q_wuji, res.valid, wuji_home(rt.kin), kp["timestamps_s"],
                                              hold.WUJI_SPEED_CAP_RAD_S)
        out |= {
            f"g1_arm_joint_names_{side}": np.array([f"{side}_{j}_joint" for j in G1_ARM_JOINTS]),
            f"{side}_valid": res.valid,
            f"{side}_q_wuji": res.q_wuji,
            f"{side}_at_limit": res.at_limit,
            f"{side}_q_g1_arm": np.full((T, 7), np.nan),
            f"{side}_ik_pos_err_m": np.full(T, np.nan),
            f"{side}_ik_rot_err_rad": np.full(T, np.nan),
            f"{side}_tip_err_m": res.tip_err_m,
            f"{side}_T_cam_from_palm": res.T_cam_from_palm,
            f"{side}_palm_rms_m": res.palm_rms_m,
            f"{side}_scale": np.array(res.scale),
            f"{side}_visible": visible,
            f"{side}_q_wuji_cmd": q_cmd,
            f"{side}_held": held,
            f"{side}_wuji_catching_up": catching,
        }
    return out, results
