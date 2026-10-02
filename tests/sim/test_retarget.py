"""B2 tests: Wuji keypoint map, palm fit, retargeter and the contract-v2 retarget output (on the stub)."""

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wuji_cap.common import derived_io, frames
from wuji_cap.common import joint_order as jo
from wuji_cap.sim import hand_keypoints as hk
from wuji_cap.sim import stub_keypoints, wuji_model as wm
from wuji_cap.sim.retarget import PALM_RMS_FAIL_M, WujiRetargeter, retarget_session


@pytest.fixture(scope="module", params=jo.SIDES)
def side(request):
    return request.param


@pytest.fixture(scope="module")
def kin(side):
    return hk.WujiKinematics(side)


@pytest.fixture(scope="module")
def retargeter(side):
    return WujiRetargeter(side)


@pytest.fixture(scope="module")
def stub_output():
    arrays, _ = stub_keypoints.make_stub()
    out, results = retarget_session(arrays)
    return arrays, out, results


def random_q(kin, rng):
    lo, hi = kin.q_lower_sdk, kin.q_upper_sdk
    return lo + (hi - lo) * (0.15 + 0.7 * rng.random(jo.N_JOINTS))


def test_keypoint_tips_match_mujoco_tip_sites(side, kin):
    """Independent check of the MediaPipe tip mapping: Pinocchio keypoints vs MuJoCo MJCF tip sites."""
    hand = wm.load_hand(side)
    data = mujoco.MjData(hand.model)
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = random_q(kin, rng)
        hand.set_q_sdk(data, q)
        mujoco.mj_kinematics(hand.model, data)
        assert np.allclose(kin.keypoints21(q)[list(hk.MP_TIPS)], hand.tips_in_wrist(data), atol=1e-6)


def test_keypoint_chain_order(side, kin):
    """Flexing one finger's PIP moves exactly that finger's PIP-distal keypoints (MediaPipe order)."""
    q0 = np.zeros(jo.N_JOINTS)
    for f_idx, f in enumerate(jo.FINGERS):
        q = q0.copy()
        q[jo.sdk_index(f, 3)] = 1.0
        moved = np.flatnonzero(np.linalg.norm(kin.keypoints21(q) - kin.keypoints21(q0), axis=1) > 1e-6)
        first = 1 + 4 * f_idx
        assert moved.tolist() == [first + 2, first + 3]


def test_palm_template_is_pose_independent(side, kin):
    rng = np.random.default_rng(1)
    for _ in range(10):
        assert np.allclose(kin.keypoints21(random_q(kin, rng))[list(hk.PALM_KP)], kin.palm_template, atol=1e-9)


def test_left_template_mirrors_right():
    left, right = hk.WujiKinematics("left").palm_template, hk.WujiKinematics("right").palm_template
    assert np.allclose(left, right * [1, -1, 1], atol=1e-3)


def test_fit_palm_recovers_pose_and_scale(side, kin):
    rng = np.random.default_rng(2)
    T = frames.make_T(Rotation.random(random_state=5).as_matrix(), [0.1, 0.2, 0.5])
    kp_cam = frames.apply_T(T, kin.keypoints21(random_q(kin, rng)) / 0.85)
    s = hk.palm_scale(kp_cam, kin.palm_template)
    fit = hk.fit_palm(kp_cam, kin.palm_template, s)
    assert np.isclose(s, 0.85) and fit.rms_m < 1e-9
    assert np.allclose(fit.T_cam_from_palm, T)


def test_palm_fit_alone_cannot_tell_sides(side, kin):
    """Finding (B2): the Wuji palm points are nearly coplanar, so the mirror-image hand fits the palm
    template almost as well as a rotation (residual of a few mm, like human noise). Side errors must be
    caught downstream (test_sides_not_swapped), not by the palm residual."""
    other = hk.WujiKinematics("left" if side == "right" else "right")
    fit = hk.fit_palm(other.keypoints21(np.zeros(jo.N_JOINTS)), kin.palm_template, 1.0)
    assert fit.rms_m < PALM_RMS_FAIL_M


def test_solve_frame_recovers_joint_angles(side, kin, retargeter):
    rng = np.random.default_rng(3)
    errs = []
    for _ in range(15):
        q = random_q(kin, rng)
        retargeter.reset()
        q_hat, ok = retargeter.solve_frame(kin.keypoints21(q))
        assert ok
        errs.append(np.abs(q_hat - q))
    errs = np.array(errs)
    assert np.median(errs) < 0.01 and errs.max() < 0.1


def test_retarget_output_matches_contract(stub_output):
    _, out, _ = stub_output
    assert derived_io.check_retarget(out) == stub_keypoints.N_FRAMES
    assert tuple(out["wuji_joint_names"]) == jo.SDK_JOINT_NAMES


def test_stub_recovery(side, stub_output):
    kp, out, results = stub_output
    v = out[f"{side}_valid"]
    assert np.array_equal(v, kp[f"{side}_valid"])
    assert np.isclose(results[side].scale, stub_keypoints.SCALE_TRUE)
    e = np.abs(out[f"{side}_q_wuji"][v] - kp[f"stub_{side}_q_true_sdk"][v])
    assert np.median(e) < 0.02 and np.percentile(e, 95) < 0.1 and e.max() < 0.25
    assert np.nanpercentile(out[f"{side}_tip_err_m"], 95) < 0.002
    # Stub wrist frame = Wuji wrist frame, so the fitted palm pose must equal it (AS-03 offset = I).
    assert np.allclose(out[f"{side}_T_cam_from_palm"][v], kp[f"{side}_T_cam_from_wrist"][v], atol=1e-9)


def test_gap_is_not_bridged(side, stub_output):
    """Invalid frames give NaN, and the first frame after a gap is solved from scratch (no warm start)."""
    kp, out, _ = stub_output
    a, b = stub_keypoints.INVALID[side]
    assert np.isnan(out[f"{side}_q_wuji"][a:b]).all() and not out[f"{side}_valid"][a:b].any()
    fresh = WujiRetargeter(side).run(kp[f"{side}_kp21_cam"][b:b + 1], np.ones(1, bool),
                                     scale=float(out[f"{side}_scale"]))
    assert np.allclose(out[f"{side}_q_wuji"][b], fresh.q_wuji[0])


def test_sides_not_swapped(stub_output):
    """Left keypoints through the right-hand retargeter: fingers curl the wrong way, so the tip error is
    an order of magnitude above the correct side's and joints pile up at limits."""
    kp, out, _ = stub_output
    res = WujiRetargeter("right").run(kp["left_kp21_cam"][:60], kp["left_valid"][:60])
    correct = out["left_tip_err_m"][:60]
    assert np.nanmedian(res.tip_err_m) > 10 * np.nanmedian(correct)
    assert np.nanmedian(res.tip_err_m) > 0.005
    assert res.at_limit.any(1).mean() > 0.5 > out["left_at_limit"][:60].any(1).mean()


def test_direction_targets_are_noop_on_wuji_fk(side, kin):
    """On Wuji's own keypoints, with lengths from the same pose, the direction targets (W1-B08)
    reproduce them exactly; lengths from the zero pose differ only through the first segment."""
    rng = np.random.default_rng(6)
    for _ in range(10):
        kp = kin.keypoints21(random_q(kin, rng))
        assert np.allclose(hk.direction_targets(kp, kp), kp, atol=1e-9)
        assert np.abs(hk.direction_targets(kp, kin.zero_kp) - kp).max() < 0.006


def test_direction_targets_keep_wuji_lengths_and_human_directions(side, kin):
    rng = np.random.default_rng(7)
    human = kin.keypoints21(random_q(kin, rng)) * 0.7 + rng.normal(scale=0.003, size=(21, 3))
    y = hk.direction_targets(human, kin.zero_kp)  # zero pose as the length reference
    assert np.allclose(y[list(hk.PALM_KP)], kin.palm_template)
    for c in hk.finger_chains():
        for a, b in zip(c[:-1], c[1:]):
            assert np.isclose(np.linalg.norm(y[b] - y[a]), np.linalg.norm(kin.zero_kp[b] - kin.zero_kp[a]))
            d_h, d_y = human[b] - human[a], y[b] - y[a]
            assert np.isclose(d_h @ d_y / np.linalg.norm(d_h) / np.linalg.norm(d_y), 1.0)


def test_bend_angles_match_joint_flexion(side, kin):
    """Flexing only a PIP joint by x changes only that finger's middle bend, by x (rotation axis is
    perpendicular to both segments at zero abduction)."""
    q = np.zeros(jo.N_JOINTS)
    q[jo.sdk_index("index", 3)] = 0.6
    d = hk.bend_angles(kin.keypoints21(q)) - hk.bend_angles(kin.zero_kp)
    assert np.isclose(d[1, 1], 0.6, atol=0.02)
    d[1, 1] = 0.0
    assert np.abs(d).max() < 1e-6


def test_stub_bend_angles_match(side, stub_output):
    """Independent check on the stub. (1) The metric is exact: the true Wuji pose has the human (stub)
    bend angles. (2) The retargeted pose matches them: median < 1 deg; p95 < 7 deg, because the
    norm_delta smoothing (W1-B06) lags the distal joints by up to ~0.07-0.1 rad on this trajectory
    (first threshold, 3 deg, was set before measuring and did not allow for that known lag)."""
    kp, out, results = stub_output
    v = out[f"{side}_valid"]
    kin = hk.WujiKinematics(side)
    human = hk.bend_angles(kp[f"{side}_kp21_cam"][v])
    true = hk.bend_angles(np.array([kin.keypoints21(q) for q in kp[f"stub_{side}_q_true_sdk"][v]]))
    assert np.allclose(true, human, atol=1e-6)
    wuji = hk.bend_angles(np.array([kin.keypoints21(q) for q in out[f"{side}_q_wuji"][v]]))
    err = np.degrees(np.abs(wuji - human))
    assert np.median(err) < 1.0 and np.percentile(err, 95) < 7.0


def test_direction_mode_handles_short_human_fingers(side, kin):
    """W1-B08 regression: a hand whose finger segments are 0.6x Wuji's (palm unchanged), like
    session_032 after the palm scale. Direction targets recover the joint angles; the superseded
    scaled targets (W1-B05) pin joints at limits instead."""
    t = np.arange(40) / 30.0
    q_true = stub_keypoints.joint_trajectory(kin, t, seed=11)
    kp = []
    for q in q_true:
        x = kin.keypoints21(q)
        y = x.copy()
        for c in hk.finger_chains():
            for a, b in zip(c[:-1], c[1:]):
                y[b] = y[a] + 0.6 * (x[b] - x[a])
        kp.append(y)
    kp = np.array(kp)
    valid = np.ones(len(t), bool)
    direction = WujiRetargeter(side).run(kp, valid, scale=1.0)
    scaled = WujiRetargeter(side, target_mode="scaled").run(kp, valid, scale=1.0)
    e_dir = np.abs(direction.q_wuji - q_true)
    assert np.median(e_dir) < 0.02 and np.percentile(e_dir, 95) < 0.12
    assert np.median(np.abs(scaled.q_wuji - q_true)) > 3 * np.median(e_dir)
    assert scaled.at_limit.any(1).mean() > 0.5 > direction.at_limit.any(1).mean()
