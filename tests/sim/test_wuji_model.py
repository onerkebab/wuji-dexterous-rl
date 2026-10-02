"""Physical joint-order checks on the Beta 2 MJCF (MuJoCo) and URDF (Pinocchio, as used by wuji-retargeting).

These catch the ordering bugs that name bookkeeping alone cannot: a finger swap, a flex/abd swap,
a proximal/distal swap, a sign flip, and the Pinocchio (alphabetical-by-finger) DoF order.
"""

import mujoco
import numpy as np
import pinocchio as pin
import pytest

from wuji_cap.common import joint_order as jo
from wuji_cap.sim import wuji_model as wm

# Spec section 2 "Wuji joint ranges" (degrees), per SDK joint1..4; mirrored left/right (AS-10).
SPEC_RANGE_DEG = {
    "thumb": [(-68, 74), (-85, 40), (-60, 90), (-60, 90)],
    "finger": [(-60, 90), (-40, 40), (-60, 120), (-60, 90)],
}
FLEX_JOINTS = (1, 3, 4)  # SDK joint numbers that flex (joint2 is abduction)


@pytest.fixture(scope="module", params=jo.SIDES)
def hand(request):
    return wm.load_hand(request.param)


def _fk(hand, q):
    d = mujoco.MjData(hand.model)
    hand.set_q_sdk(d, q)
    mujoco.mj_kinematics(hand.model, d)
    return d


def test_wuji_description_pinned():
    commit = wm.wuji_description_commit()
    assert commit.startswith(wm.WUJI_DESCRIPTION_COMMIT), commit
    assert not commit.endswith("-dirty"), commit


def test_mjcf_file_order_is_sdk_order(hand):
    # TWIST2's wuji_hand_bridge assumes MJCF joint order == device order; guard that assumption.
    assert wm.mjcf_joint_names_in_file_order(hand.model) == list(jo.model_joint_names(hand.side))
    assert hand.model.nq == hand.model.nu == 20


def test_limits_match_spec(hand):
    rng_deg = np.degrees(hand.joint_range)
    for i, name in enumerate(jo.SDK_JOINT_NAMES):
        finger, j = name.split("_joint")
        expected = SPEC_RANGE_DEG["thumb" if finger == "thumb" else "finger"][int(j) - 1]
        np.testing.assert_allclose(rng_deg[i], expected, atol=0.5, err_msg=f"SDK {i} {name}")


def test_actuator_ctrlrange_matches_joint_range(hand):
    np.testing.assert_array_equal(hand.model.actuator_ctrlrange[hand.act_ids], hand.joint_range)


def test_left_right_mirrored_limits():
    np.testing.assert_array_equal(wm.load_hand("left").joint_range, wm.load_hand("right").joint_range)


def test_single_joint_moves_only_its_finger_and_distal_bodies(hand):
    d0 = _fk(hand, np.zeros(20))
    tips0 = hand.tips_in_wrist(d0)
    for i, name in enumerate(jo.SDK_JOINT_NAMES):
        f, j = i // 4, i % 4 + 1
        q = np.zeros(20)
        q[i] = 0.3
        d = _fk(hand, q)
        moved_tip = np.linalg.norm(hand.tips_in_wrist(d) - tips0, axis=1)
        assert moved_tip[f] > 5e-3, (name, moved_tip)
        assert np.all(np.delete(moved_tip, f) < 1e-9), (name, moved_tip)
        # joint1 (most proximal) moves 5 finger bodies, ..., joint4 (most distal) moves 2.
        moved = np.linalg.norm(d.xpos - d0.xpos, axis=1) + np.linalg.norm(d.xquat - d0.xquat, axis=1) > 1e-9
        moved_names = [hand.model.body(b).name for b in np.flatnonzero(moved)]
        assert len(moved_names) == 6 - j, (name, moved_names)
        assert all(jo.FINGERS[f] in n for n in moved_names), (name, moved_names)


def _palm_axes(hand, d):
    """Unit knuckle line (index -> pinky) and a palm-plane normal, in the wrist frame."""
    p = jo.side_prefix(hand.side)
    R, w = d.xmat[hand.wrist_body_id].reshape(3, 3), d.xpos[hand.wrist_body_id]
    kn = {b: (d.xpos[hand.model.body(p + b).id] - w) @ R for b in
          ("index_finger_proximal", "middle_finger_proximal", "pinky_proximal")}
    k = kn["pinky_proximal"] - kn["index_finger_proximal"]
    k /= np.linalg.norm(k)
    n = np.cross(k, kn["middle_finger_proximal"])
    return k, n / np.linalg.norm(n)


def test_flex_vs_abd_direction_and_sign(hand):
    d0 = _fk(hand, np.zeros(20))
    tips0 = hand.tips_in_wrist(d0)
    k, n = _palm_axes(hand, d0)
    # Heuristic palm side: the thumb tip sits on the palm side of the knuckle/wrist plane at zero pose.
    palm_side = np.sign(tips0[0] @ n)
    assert abs(tips0[0] @ n) > 5e-3
    for f in range(1, 5):  # index..pinky
        for j in (1, 2, 3, 4):
            q = np.zeros(20)
            q[4 * f + j - 1] = 0.3
            disp = hand.tips_in_wrist(_fk(hand, q))[f] - tips0[f]
            disp /= np.linalg.norm(disp)
            name = jo.SDK_JOINT_NAMES[4 * f + j - 1]
            if j in FLEX_JOINTS:  # positive flex/pip/dip curls the finger toward the palm
                assert disp @ n * palm_side > 0.9, (name, disp @ n)
            else:  # abduction moves the tip sideways, along the knuckle line
                assert abs(disp @ k) > 0.9, (name, disp @ k)


def test_thumb_flex_joints_curl_same_side(hand):
    d0 = _fk(hand, np.zeros(20))
    tip0 = hand.tips_in_wrist(d0)[0]
    _, n = _palm_axes(hand, d0)
    signs = []
    for j in FLEX_JOINTS:
        q = np.zeros(20)
        q[j - 1] = 0.3
        signs.append(np.sign((hand.tips_in_wrist(_fk(hand, q))[0] - tip0) @ n))
    assert len(set(signs)) == 1, signs


# --- Pinocchio / URDF (the model wuji-retargeting uses) ---

def _pin(side):
    model = pin.buildModelFromUrdf(str(wm.urdf_path(side)))
    names = [n for i, n in enumerate(model.names) if model.nqs[i] > 0]  # == Robot.dof_joint_names
    return model, names


def _pin_tips_in_wrist(model, data, q):
    pin.framesForwardKinematics(model, data, q)
    side = "left" if model.name.endswith("left") else "right"
    W = data.oMf[model.getFrameId(jo.side_prefix(side) + "wrist")].inverse()
    return np.array([(W * data.oMf[model.getFrameId(t)]).translation for t in jo.tip_names(side)])


@pytest.mark.parametrize("side", jo.SIDES)
def test_pinocchio_dof_order_is_qpos_order_and_not_sdk(side):
    model, names = _pin(side)
    assert [model.idx_qs[model.getJointId(n)] for n in names] == list(range(20))
    # Pinocchio sorts fingers alphabetically; this is why every conversion goes through joint_order.
    assert names != list(jo.model_joint_names(side))
    assert set(names) == set(jo.model_joint_names(side))


@pytest.mark.parametrize("side", jo.SIDES)
def test_urdf_limits_match_mjcf(side):
    model, names = _pin(side)
    lim = np.stack([model.lowerPositionLimit, model.upperPositionLimit], axis=-1)  # [20, 2] pin order
    np.testing.assert_allclose(jo.to_sdk(lim.T, names, side).T, wm.load_hand(side).joint_range, atol=1e-6)


@pytest.mark.parametrize("side", jo.SIDES)
def test_cross_library_fingertip_fk(side):
    """MuJoCo (MJCF) and Pinocchio (URDF) fingertips agree within 1 mm (M0 exit criterion)."""
    hand = wm.load_hand(side)
    model, names = _pin(side)
    data = model.createData()
    rng = np.random.default_rng(0)
    lo, hi = hand.joint_range.T
    worst, worst_unmapped = 0.0, np.inf
    for _ in range(100):
        q_sdk = rng.uniform(lo, hi)
        tips_mj = hand.tips_in_wrist(_fk(hand, q_sdk))
        err = np.linalg.norm(_pin_tips_in_wrist(model, data, jo.from_sdk(q_sdk, names, side)) - tips_mj, axis=1)
        worst = max(worst, err.max())
        # Negative control: feeding SDK order straight to Pinocchio must be caught.
        err_bad = np.linalg.norm(_pin_tips_in_wrist(model, data, q_sdk) - tips_mj, axis=1)
        worst_unmapped = min(worst_unmapped, err_bad.max())
    assert worst < 1e-3, worst
    assert worst_unmapped > 1e-2, worst_unmapped
