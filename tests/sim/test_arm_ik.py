import numpy as np
import pytest

from wuji_cap.common import frames
from wuji_cap.common import joint_order as jo
from wuji_cap.sim import g1_model
from wuji_cap.sim.arm_ik import POS_TOL_M, ROT_TOL_RAD, ArmIK

pytestmark = pytest.mark.skipif(not g1_model.g1_xml_path().is_file(), reason="TWIST2 G1 model not found")


@pytest.fixture(scope="module")
def ik():
    return ArmIK()


def fk_palm_cam(ik: ArmIK, q_arm: dict) -> dict:
    """Palm poses in the camera frame for given arm angles (what B2 would hand to B3)."""
    g = ik.g
    q = g.q_home.copy()
    for s, qa in q_arm.items():
        q[g.arm_qpos_adr[s]] = qa
    ik.config.update(q)
    return {s: frames.inv_T(ik.T_world_from_cam) @ g.body_T(g.palm_body_ids[s], ik.config.data) for s in jo.SIDES}


def reachable_arm_poses(ik: ArmIK, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append({s: rng.uniform(*(ik.g.model.jnt_range[ik.g.arm_jnt_ids[s]].T * 0.5))
                    for s in jo.SIDES})
    return out


def test_round_trip_reaches_fk_targets(ik):
    """Targets made by FK from known arm poses are reached to the solver tolerance, frame after frame."""
    poses = reachable_arm_poses(ik, 6, seed=1)
    targets = [fk_palm_cam(ik, p) for p in poses]
    T_cam = {s: np.array([t[s] for t in targets]) for s in jo.SIDES}
    res = ik.run(T_cam, {s: np.ones(len(poses), bool) for s in jo.SIDES})
    for s in jo.SIDES:
        assert (res.pos_err_m[s] < POS_TOL_M).all(), res.pos_err_m[s]
        assert (res.rot_err_rad[s] < ROT_TOL_RAD).all(), res.rot_err_rad[s]
        lo, hi = ik.g.model.jnt_range[ik.g.arm_jnt_ids[s]].T
        assert ((res.q_arm[s] >= lo - 1e-6) & (res.q_arm[s] <= hi + 1e-6)).all()


def test_only_arms_move(ik):
    target = fk_palm_cam(ik, reachable_arm_poses(ik, 1, seed=2)[0])
    res = ik.run({s: target[s][None] for s in jo.SIDES}, {s: np.ones(1, bool) for s in jo.SIDES})
    assert np.isfinite(res.q_arm["left"]).all()
    q = ik.config.q
    arm = np.concatenate([ik.g.arm_qpos_adr[s] for s in jo.SIDES])
    others = np.setdiff1d(np.arange(ik.g.model.nq), arm)
    assert np.allclose(q[others], ik.g.q_home[others], atol=1e-9)


def test_unreachable_and_invalid_frames(ik):
    """A target 2 m in front of the camera is reported as a large error (within limits); an invalid
    frame gives NaN for that side only."""
    far = frames.make_T(np.eye(3), np.array([0.0, 0.0, 2.0]))
    near = fk_palm_cam(ik, reachable_arm_poses(ik, 1, seed=3)[0])
    T_cam = {"left": np.stack([far, near["left"]]), "right": np.stack([near["right"], near["right"]])}
    valid = {"left": np.array([True, True]), "right": np.array([False, True])}
    res = ik.run(T_cam, valid)
    assert res.pos_err_m["left"][0] > 0.5
    lo, hi = ik.g.model.jnt_range[ik.g.arm_jnt_ids["left"]].T
    assert ((res.q_arm["left"][0] >= lo - 1e-6) & (res.q_arm["left"][0] <= hi + 1e-6)).all()
    assert res.pos_err_m["left"][1] < POS_TOL_M  # recovers after the failure (restart)
    assert np.isnan(res.q_arm["right"][0]).all() and np.isnan(res.pos_err_m["right"][0])
    assert res.pos_err_m["right"][1] < POS_TOL_M


def test_deterministic(ik):
    poses = reachable_arm_poses(ik, 3, seed=4)
    T_cam = {s: np.array([fk_palm_cam(ik, p)[s] for p in poses]) for s in jo.SIDES}
    T_cam["left"][1] = frames.make_T(np.eye(3), np.array([0.0, 0.0, 2.0]))  # forces the restart path
    v = {s: np.ones(3, bool) for s in jo.SIDES}
    a, b = ik.run(T_cam, v), ik.run(T_cam, v)
    for s in jo.SIDES:
        assert np.array_equal(a.q_arm[s], b.q_arm[s])


def test_free_waist_stays_upright_when_arms_reach(ik):
    poses = reachable_arm_poses(ik, 3, seed=5)
    T_cam = {s: np.array([fk_palm_cam(ik, p)[s] for p in poses]) for s in jo.SIDES}
    out = ik.run_free_waist(T_cam, {s: np.ones(3, bool) for s in jo.SIDES})
    assert np.allclose(out["q_waist"], ik.g.q_home[ik.g.waist_qpos_adr])
    for s in jo.SIDES:
        assert (out[s]["pos_err_m"] < POS_TOL_M).all()


def test_free_waist_recovers_a_lean_only_target(ik):
    """A palm pose made by FK with the torso pitched forward and the arms stretched is out of reach upright
    but reached with the waist free (the AS-B08 diagnostic)."""
    g = ik.g
    q = g.q_home.copy()
    q[g.waist_qpos_adr] = [0.0, 0.0, 0.45]
    for s in jo.SIDES:
        q[g.arm_qpos_adr[s]] = [-1.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ik.config.update(q)
    T_cam = {s: (frames.inv_T(ik.T_world_from_cam) @ g.body_T(g.palm_body_ids[s], ik.config.data))[None] for s in jo.SIDES}
    valid = {s: np.ones(1, bool) for s in jo.SIDES}
    fixed = ik.run(T_cam, valid)
    free = ik.run_free_waist(T_cam, valid)
    for s in jo.SIDES:
        assert fixed.pos_err_m[s][0] > 0.02
        assert free[s]["pos_err_m"][0] < 1e-3
    assert free["q_waist"][0, 2] > 0.2


def test_converged_restart_beats_hysteresis(ik):
    """Regression: a restart that converges must replace a warm result stuck at ~1 mm, even though the
    gain is below the 5 mm hysteresis (seed 5, right arm, frame 1 stopped at 1.29 mm before the fix)."""
    poses = reachable_arm_poses(ik, 3, seed=5)
    T_cam = {s: np.array([fk_palm_cam(ik, p)[s] for p in poses]) for s in jo.SIDES}
    res = ik.run(T_cam, {s: np.ones(3, bool) for s in jo.SIDES})
    for s in jo.SIDES:
        assert (res.pos_err_m[s] < POS_TOL_M).all(), res.pos_err_m[s]


def interp_poses(a: dict, b: dict, n: int) -> list[dict]:
    return [{s: (1 - x) * a[s] + x * b[s] for s in jo.SIDES} for x in np.linspace(0, 1, n)]


def test_hold_through_gap_and_resume(ik):
    """B4 (W1-B14): invalid frames give NaN contract fields and hold the arm (home before the first valid
    frame, then the last solved pose); the frame after the gap warm-starts from the held pose and converges.
    Before B4 an invalid frame reset the arm to home."""
    a, b = reachable_arm_poses(ik, 2, seed=7)
    poses = interp_poses(a, b, 12)
    T_cam = {s: np.array([fk_palm_cam(ik, p)[s] for p in poses]) for s in jo.SIDES}
    valid = {s: np.ones(12, bool) for s in jo.SIDES}
    valid["left"][[0, 1, 5, 6, 7]] = False
    res = ik.run(T_cam, valid)
    adr = ik.g.arm_qpos_adr["left"]
    assert np.array_equal(res.held["left"], ~valid["left"]) and not res.held["right"].any()
    assert np.isnan(res.q_arm["left"][~valid["left"]]).all() and np.isnan(res.pos_err_m["left"][~valid["left"]]).all()
    assert np.allclose(res.q_arm_hold["left"][[0, 1]], ik.g.q_home[adr])
    assert np.array_equal(res.q_arm_hold["left"][5:8], np.repeat(res.q_arm["left"][4][None], 3, 0))
    assert np.array_equal(res.q_arm_hold["left"][valid["left"]], res.q_arm["left"][valid["left"]])
    assert (res.pos_err_m["left"][valid["left"]] < POS_TOL_M).all()
    assert res.start_used["left"][8] == 0  # warm start from the held pose, no restart needed


def test_no_arm_jump_at_gap_edges(ik):
    """B4 test 5: the hand leaves and comes back where it left. Playback steps at the gap edges are no larger
    than the largest step between valid frames; a reset to home (the pre-B4 behaviour) would jump."""
    a, b = reachable_arm_poses(ik, 2, seed=8)
    poses = interp_poses(a, b, 10)
    poses = poses + poses[::-1]  # out and back: frames 8..11 (around the turn) are unseen
    T_cam = {s: np.array([fk_palm_cam(ik, p)[s] for p in poses]) for s in jo.SIDES}
    valid = {s: np.ones(len(poses), bool) for s in jo.SIDES}
    for s in jo.SIDES:
        valid[s][8:12] = False
    res = ik.run(T_cam, valid)
    for s in jo.SIDES:
        step = np.r_[0.0, np.abs(np.diff(res.q_arm_hold[s], axis=0)).max(1)]
        usable = ~res.held[s]
        gap_end = usable & np.r_[False, res.held[s][:-1]]
        within = usable & np.r_[False, usable[:-1]]
        assert gap_end.sum() == 1
        assert step[gap_end].max() <= step[within].max() + 1e-6
        home_step = np.abs(res.q_arm[s][12] - ik.g.q_home[ik.g.arm_qpos_adr[s]]).max()
        assert home_step > step[within].max()  # a reset to home (pre-B4) would break the criterion above
