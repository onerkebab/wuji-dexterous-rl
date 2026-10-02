"""Pure joint-order tests: SDK names, permutation algebra, round trips (no model files needed)."""

import numpy as np
import pytest

from wuji_cap.common import joint_order as jo


def test_sdk_names_match_sdk_labels():
    # WujiHandJointHandle.label: "0 thumb_joint1, 1 thumb_joint2, ..., 19 pinky_joint4" (wuji_sdk stubs).
    assert len(jo.SDK_JOINT_NAMES) == 20
    assert jo.SDK_JOINT_NAMES[0] == "thumb_joint1"
    assert jo.SDK_JOINT_NAMES[1] == "thumb_joint2"
    assert jo.SDK_JOINT_NAMES[4] == "index_joint1"
    assert jo.SDK_JOINT_NAMES[19] == "pinky_joint4"
    for i, name in enumerate(jo.SDK_JOINT_NAMES):
        finger, joint = name.split("_joint")
        assert jo.sdk_index(finger, int(joint)) == i


@pytest.mark.parametrize("side", jo.SIDES)
def test_model_names_finger_major(side):
    names = jo.model_joint_names(side)
    acts = jo.mjcf_actuator_names(side)
    assert len(names) == len(set(names)) == 20
    assert len(acts) == len(set(acts)) == 20
    p = jo.side_prefix(side)
    for i, (sdk, n, a) in enumerate(zip(jo.SDK_JOINT_NAMES, names, acts)):
        finger, j = sdk.split("_joint")
        assert n.startswith(p + finger), (i, sdk, n)
        assert a.endswith(f"J{int(j) - 1}"), (i, sdk, a)
    # Spot checks against the Beta 2 MJCF, written out by hand on purpose.
    assert names[0] == p + "thumb_cmc_flex"
    assert names[5] == p + "index_finger_mcp_abd"
    assert names[14] == p + "ring_finger_pip"
    assert names[19] == p + "pinky_dip"
    assert acts[0] == p + "THJ0" and acts[5] == p + "FFJ1" and acts[19] == p + "LFJ3"


def test_side_names_are_disjoint():
    assert not set(jo.model_joint_names("left")) & set(jo.model_joint_names("right"))
    with pytest.raises(ValueError):
        jo.side_prefix("both")


def test_permutation_semantics():
    src = ["a", "b", "c"]
    dst = ["c", "a", "b"]
    perm = jo.permutation(src, dst)
    x_src = np.array([10.0, 20.0, 30.0])
    assert x_src[perm].tolist() == [30.0, 10.0, 20.0]


def test_permutation_rejects_bad_name_sets():
    with pytest.raises(ValueError):
        jo.permutation(["a", "b"], ["a", "c"])
    with pytest.raises(ValueError):
        jo.permutation(["a", "a"], ["a", "a"])
    with pytest.raises(ValueError):
        jo.permutation(["a", "b"], ["a"])


@pytest.mark.parametrize("side", jo.SIDES)
def test_round_trip_random_orderings(side):
    rng = np.random.default_rng(0)
    sdk_names = jo.model_joint_names(side)
    for _ in range(20):
        other = list(rng.permutation(sdk_names))
        q_sdk = rng.normal(size=(7, 20))  # batch dims are preserved
        q_other = jo.from_sdk(q_sdk, other, side)
        for k, name in enumerate(other):  # element-wise meaning, not just a round trip
            np.testing.assert_array_equal(q_other[:, k], q_sdk[:, sdk_names.index(name)])
        np.testing.assert_array_equal(jo.to_sdk(q_other, other, side), q_sdk)


def test_identity_for_sdk_order():
    names = jo.model_joint_names("right")
    np.testing.assert_array_equal(jo.permutation(names, names), np.arange(20))


def test_wrong_length_rejected():
    with pytest.raises(ValueError):
        jo.from_sdk(np.zeros(19), jo.model_joint_names("right"), "right")
    with pytest.raises(ValueError):
        jo.to_sdk(np.zeros(20), jo.model_joint_names("left"), "right")  # wrong side's names
