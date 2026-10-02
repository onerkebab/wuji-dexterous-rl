"""B3: G1 arm IK (mink) so each Wuji palm reaches its retargeted pose.

Target per frame and side: ``T_world_from_palm = T_world_from_torso @ T_torso_from_cam @ T_cam_from_palm``
(frames.T_torso_from_cam, AS-01; palm = Wuji wrist body, W1-B05). One differential-IK QP per iteration
(mink + daqp) per arm with its palm FrameTask, a weak posture task toward the home pose (picks the elbow in the
redundant direction), joint limits as constraints, and every non-arm DoF frozen (AS-09). Each frame
warm-starts from the previous solution; if that does not converge, the arm is re-solved from the home pose
and from fixed seeded random poses, and the result closest in position is kept (see solve_side) (warm starts alone drift into joint-limit traps: session_032 left arm,
median 43 mm error, while every sampled frame is exactly reachable from home).

B4 (W1-B14): an invalid frame (not usable: out of view or not retargeted) holds that arm where it is; the next
valid frame warm-starts from the held pose. ``q_arm`` stays NaN there; ``q_arm_hold`` is the held pose (home
before the first valid frame). The playback array with the speed-capped return is made from ``q_arm`` by
``hold.playback`` (scripts/b3_arm_ik.py).

Errors are measured on the final pose with MuJoCo FK, not taken from the solver.
Design W1-B10 and reach-failure threshold W1-B11 (both approved 2026-10-01).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mink
import mujoco
import numpy as np

from wuji_cap.common import frames
from wuji_cap.common import joint_order as jo
from wuji_cap.sim.g1_model import G1Wuji

POSITION_COST = 1.0  # per m
ORIENTATION_COST = 0.2  # per rad: 1 rad of palm rotation weighs like 0.2 m of position (W1-B10)
POSTURE_COST = 1e-3
LM_DAMPING = 1e-3
MAX_ITERS = 60
POS_TOL_M = 1e-4
ROT_TOL_RAD = 1e-3
STEP_TOL_RAD = 1e-6
SELECT_ROT_WEIGHT = 0.02  # m per rad, choosing among restarts: position first (best-effort reach)
SELECT_HYSTERESIS_M = 0.005
N_SEEDS = 8  # fixed random restart poses per arm (seeded, so runs are deterministic)
REACH_FAIL_POS_M = 0.01  # W1-B11 (approved 2026-10-01): reach failure if position error > 10 mm
REACH_FAIL_ROT_RAD = float(np.radians(5.0))  # ... or rotation error > 5 deg
ARM_AT_LIMIT_TOL_RAD = 0.02  # same provisional tolerance as the hand (W1-17)


@dataclass
class IKResult:
    q_arm: dict  # side -> [T, 7]
    pos_err_m: dict  # side -> [T]
    rot_err_rad: dict  # side -> [T]
    n_iters: dict  # side -> [T] int
    start_used: dict  # side -> [T] int: 0 warm start, 1 home pose, 2+ fixed random seed (see solve_side)
    at_arm_limit: dict  # side -> [T, 7] bool
    q_arm_hold: dict  # side -> [T, 7]: q_arm on valid frames, the held pose otherwise (B4; no return ramp)
    held: dict  # side -> [T] bool: not solved, arm held (B4)
    T_world_from_palm_target: dict  # side -> [T, 4, 4]
    params: dict = field(default_factory=dict)


def palm_errors(g: G1Wuji, data: mujoco.MjData, side: str, T_target: np.ndarray) -> tuple[float, float]:
    T = g.body_T(g.palm_body_ids[side], data)
    return float(np.linalg.norm(T[:3, 3] - T_target[:3, 3])), float(frames.rotation_angle(T[:3, :3].T @ T_target[:3, :3]))


def _se3(T: np.ndarray) -> mink.SE3:
    return mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(T[:3, :3]), T[:3, 3])


class ArmIK:
    def __init__(self, g: G1Wuji | None = None, position_cost: float = POSITION_COST,
                 orientation_cost: float = ORIENTATION_COST, posture_cost: float = POSTURE_COST,
                 T_world_from_cam: np.ndarray | None = None):
        self.g = g if g is not None else G1Wuji()
        m = self.g.model
        self.config = mink.Configuration(m, self.g.q_home)
        self.palm_tasks = {
            s: mink.FrameTask(m.body(self.g.palm_body_ids[s]).name, "body", position_cost=position_cost,
                              orientation_cost=orientation_cost, lm_damping=LM_DAMPING)
            for s in jo.SIDES
        }
        self.posture = mink.PostureTask(m, cost=posture_cost)
        self.posture.set_target(self.g.q_home)
        self.limits = [mink.ConfigurationLimit(m)]
        self.T_world_from_cam = (self.g.T_world_from_torso() @ frames.T_torso_from_cam()
                                 if T_world_from_cam is None else np.asarray(T_world_from_cam, float).copy())
        T_ref = self.T_world_from_cam
        if (T_ref.shape != (4, 4) or not np.isfinite(T_ref).all()
                or not np.allclose(T_ref[3], [0, 0, 0, 1])
                or not np.allclose(T_ref[:3, :3].T @ T_ref[:3, :3], np.eye(3), atol=1e-8)
                or not np.isclose(np.linalg.det(T_ref[:3, :3]), 1.0)):
            raise ValueError("T_world_from_cam must be a finite rigid transform")
        rng = np.random.default_rng(0)
        self.seeds = {}
        for s in jo.SIDES:
            lo, hi = m.jnt_range[self.g.arm_jnt_ids[s]].T
            self.seeds[s] = rng.uniform(lo + 0.1, hi - 0.1, size=(N_SEEDS, 7))
        self.params = {"position_cost": position_cost, "orientation_cost": orientation_cost,
                       "posture_cost": posture_cost, "lm_damping": LM_DAMPING, "max_iters": MAX_ITERS,
                       "pos_tol_m": POS_TOL_M, "n_seeds": N_SEEDS, "select_rot_weight_m_per_rad": SELECT_ROT_WEIGHT, "select_hysteresis_m": SELECT_HYSTERESIS_M, "seed": 0, "rot_tol_rad": ROT_TOL_RAD, "solver": "daqp",
                       "posture_target": "home keyframe", "T_world_from_cam": self.T_world_from_cam.tolist()}

    def solve(self, targets: dict[str, np.ndarray | None], free_waist: bool = False) -> int:
        """Iterate from the current configuration toward the given world-frame palm targets (None: that arm
        is frozen in place). ``free_waist`` also lets the 3 waist joints move (AS-B08 diagnostic).
        Returns the number of iterations."""
        active = [s for s, T in targets.items() if T is not None]
        for s in active:
            self.palm_tasks[s].set_target(_se3(targets[s]))
        frozen_body = self.g.frozen_dofs()
        if free_waist:
            frozen_body = np.setdiff1d(frozen_body, self.g.waist_dof_adr)
        frozen = [frozen_body] + [self.g.arm_dof_adr[s] for s in jo.SIDES if s not in active]
        constraints = [mink.DofFreezingTask(self.g.model, dof_indices=np.concatenate(frozen).tolist())]
        tasks = [self.palm_tasks[s] for s in active] + [self.posture]
        if not active:
            return 0
        dt = 1.0  # a step size; mink returns velocities, so dt=1 makes them joint increments
        for it in range(1, MAX_ITERS + 1):
            v = mink.solve_ik(self.config, tasks, dt, "daqp", limits=self.limits, constraints=constraints)
            self.config.integrate_inplace(v, dt)
            errs = [palm_errors(self.g, self.config.data, s, targets[s]) for s in active]
            if all(p < POS_TOL_M and r < ROT_TOL_RAD for p, r in errs) or np.abs(v).max() < STEP_TOL_RAD:
                return it
        return MAX_ITERS

    @staticmethod
    def _select_cost(err: tuple[float, float]) -> float:
        return err[0] + SELECT_ROT_WEIGHT * err[1]

    def solve_side(self, side: str, T_target: np.ndarray) -> tuple[int, int]:
        """Solve one arm (the other frozen) from the warm start. If that does not converge, retry from the
        home pose and from ``N_SEEDS`` fixed random arm poses. Unreachable targets are best effort: pick the
        start whose result is closest in position (``_select_cost``), and leave the warm-start result only
        for a gain above ``SELECT_HYSTERESIS_M`` (otherwise the arm flips between solutions frame to frame).
        Returns (total iterations, start used: 0 warm, 1 home, 2+ seed)."""
        other = "right" if side == "left" else "left"
        targets = {side: T_target, other: None}
        n = self.solve(targets)
        err = palm_errors(self.g, self.config.data, side, T_target)
        if err[0] < POS_TOL_M and err[1] < ROT_TOL_RAD:
            return n, 0
        best_q, best_cost, best_start = self.config.q.copy(), self._select_cost(err) - SELECT_HYSTERESIS_M, 0
        adr = self.g.arm_qpos_adr[side]
        for start, q_arm in enumerate([self.g.q_home[adr], *self.seeds[side]], start=1):
            q = best_q.copy()
            q[adr] = q_arm
            self.config.update(q)
            n += self.solve(targets)
            err = palm_errors(self.g, self.config.data, side, T_target)
            if err[0] < POS_TOL_M and err[1] < ROT_TOL_RAD:  # converged always wins, hysteresis or not
                return n, start
            if self._select_cost(err) < best_cost:
                best_q, best_cost, best_start = self.config.q.copy(), self._select_cost(err), start
        self.config.update(best_q)
        return n, best_start

    def run(self, T_cam_from_palm: dict[str, np.ndarray], valid: dict[str, np.ndarray]) -> IKResult:
        T = len(next(iter(valid.values())))
        g = self.g
        res = IKResult(
            q_arm={s: np.full((T, 7), np.nan) for s in jo.SIDES},
            pos_err_m={s: np.full(T, np.nan) for s in jo.SIDES},
            rot_err_rad={s: np.full(T, np.nan) for s in jo.SIDES},
            n_iters={s: np.zeros(T, int) for s in jo.SIDES},
            start_used={s: np.zeros(T, int) for s in jo.SIDES},
            at_arm_limit={s: np.zeros((T, 7), bool) for s in jo.SIDES},
            q_arm_hold={s: np.full((T, 7), np.nan) for s in jo.SIDES},
            held={s: np.zeros(T, bool) for s in jo.SIDES},
            T_world_from_palm_target={s: np.full((T, 4, 4), np.nan) for s in jo.SIDES},
            params=dict(self.params),
        )
        self.config.update(g.q_home)
        held_q = {s: g.q_home[g.arm_qpos_adr[s]].copy() for s in jo.SIDES}  # last solved pose (B4)
        lo = {s: g.model.jnt_range[g.arm_jnt_ids[s], 0] for s in jo.SIDES}
        hi = {s: g.model.jnt_range[g.arm_jnt_ids[s], 1] for s in jo.SIDES}
        for t in range(T):
            for s in jo.SIDES:
                if not (bool(valid[s][t]) and np.isfinite(T_cam_from_palm[s][t]).all()):
                    # Hold (B4). Restored exactly: solving the other arm leaves ~1e-16 rad on the frozen one.
                    res.held[s][t] = True
                    res.q_arm_hold[s][t] = held_q[s]
                    q = self.config.q.copy()
                    q[g.arm_qpos_adr[s]] = held_q[s]
                    self.config.update(q)
                    continue
                target = self.T_world_from_cam @ T_cam_from_palm[s][t]
                res.T_world_from_palm_target[s][t] = target
                res.n_iters[s][t], res.start_used[s][t] = self.solve_side(s, target)
                q = self.config.q[g.arm_qpos_adr[s]]
                res.q_arm[s][t] = res.q_arm_hold[s][t] = held_q[s] = q
                res.pos_err_m[s][t], res.rot_err_rad[s][t] = palm_errors(g, self.config.data, s, target)
                res.at_arm_limit[s][t] = (q <= lo[s] + ARM_AT_LIMIT_TOL_RAD) | (q >= hi[s] - ARM_AT_LIMIT_TOL_RAD)
        return res

    def run_free_waist(self, T_cam_from_palm: dict[str, np.ndarray], valid: dict[str, np.ndarray]) -> dict:
        """AS-B08 diagnostic: the same palm targets, fixed in the world where the home-pose head camera puts
        them. Each frame first tries the upright waist (home pose, ``solve_side`` with its restarts); only
        if that leaves a reach failure (W1-B11) are both arms and the 3 waist joints solved together (the waist is shared), so the waist leans only when
        reach needs it. The head camera rides on the torso, so this does not model the camera moving
        with the lean; it shows how much of the reach shortfall a forward lean could recover. Same
        restart rule as ``solve_side``, applied to both arms together (sum of the per-side costs).

        Returns {side: {"pos_err_m": [T], "rot_err_rad": [T]}, "q_waist": [T, 3]}.
        """
        T = len(next(iter(valid.values())))
        g = self.g
        out = {s: {"pos_err_m": np.full(T, np.nan), "rot_err_rad": np.full(T, np.nan)} for s in jo.SIDES}
        out["q_waist"] = np.full((T, 3), np.nan)
        self.config.update(g.q_home)

        def cost(targets):
            return sum(self._select_cost(palm_errors(g, self.config.data, s, Tt)) for s, Tt in targets.items() if Tt is not None)

        def converged(targets):
            return all(e[0] < POS_TOL_M and e[1] < ROT_TOL_RAD
                       for e in (palm_errors(g, self.config.data, s, Tt) for s, Tt in targets.items() if Tt is not None))

        for t in range(T):
            targets = {}
            for s in jo.SIDES:
                ok = bool(valid[s][t]) and np.isfinite(T_cam_from_palm[s][t]).all()
                targets[s] = self.T_world_from_cam @ T_cam_from_palm[s][t] if ok else None  # None: arm held
            if all(Tt is None for Tt in targets.values()):
                continue
            q = self.config.q.copy()
            q[g.waist_qpos_adr] = g.q_home[g.waist_qpos_adr]
            self.config.update(q)
            for s, Tt in targets.items():
                if Tt is not None:
                    self.solve_side(s, Tt)  # upright, with the fixed-waist restarts
            if not any(reach_failure(*palm_errors(g, self.config.data, s, Tt)) for s, Tt in targets.items() if Tt is not None):
                out["q_waist"][t] = self.config.q[g.waist_qpos_adr]
                for s, Tt in targets.items():
                    if Tt is not None:
                        out[s]["pos_err_m"][t], out[s]["rot_err_rad"][t] = palm_errors(g, self.config.data, s, Tt)
                continue
            self.solve(targets, free_waist=True)
            if not converged(targets):
                best_q, best_cost = self.config.q.copy(), cost(targets) - SELECT_HYSTERESIS_M
                starts = [{s: g.q_home[g.arm_qpos_adr[s]] for s in jo.SIDES}]
                starts += [{s: self.seeds[s][k] for s in jo.SIDES} for k in range(N_SEEDS)]
                for start in starts:
                    q = g.q_home.copy()
                    for s in jo.SIDES:
                        q[g.arm_qpos_adr[s]] = start[s] if targets[s] is not None else best_q[g.arm_qpos_adr[s]]
                    self.config.update(q)
                    self.solve(targets, free_waist=True)
                    if converged(targets):
                        best_q = self.config.q.copy()
                        break
                    if cost(targets) < best_cost:
                        best_q, best_cost = self.config.q.copy(), cost(targets)
                self.config.update(best_q)
            out["q_waist"][t] = self.config.q[g.waist_qpos_adr]
            for s, Tt in targets.items():
                if Tt is not None:
                    out[s]["pos_err_m"][t], out[s]["rot_err_rad"][t] = palm_errors(g, self.config.data, s, Tt)
        return out


def reach_failure(pos_err_m: np.ndarray, rot_err_rad: np.ndarray) -> np.ndarray:
    """[T] bool, W1-B11. NaN (invalid) frames are not failures."""
    with np.errstate(invalid="ignore"):
        return (pos_err_m > REACH_FAIL_POS_M) | (rot_err_rad > REACH_FAIL_ROT_RAD)
