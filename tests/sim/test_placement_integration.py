"""B integration of the W1-A06 placement: the robot the batch, stills and triptych use."""
import mujoco
import numpy as np
import pytest

from wuji_cap.common import frames
from wuji_cap.common.joint_order import SIDES
from wuji_cap.sim.g1_model import HEAD_CAMERA, G1Wuji, g1_xml_path
from wuji_cap.sim.placement import (LEG_PITCH_RAD, PELVIS_OFFSET_M, move_camera_to_reference, placed_g1,
                                    reference_sole_height, reference_T_world_from_cam)

pytestmark = pytest.mark.skipif(not g1_xml_path().is_file(), reason="TWIST2 G1 model not found")


def test_unplaced_is_the_pre_b5_robot():
    np.testing.assert_array_equal(placed_g1(placed=False).q_home, G1Wuji().q_home)


def test_placed_robot_stands_on_the_floor_within_limits():
    g, base = placed_g1(), G1Wuji()
    assert abs(reference_sole_height(g)) < 1e-6  # feet on the z=0 floor (home keyframe: 0.22 m above it)
    np.testing.assert_allclose(g.q_home[:3] - base.q_home[:3], PELVIS_OFFSET_M, atol=1e-15)
    np.testing.assert_array_equal(g.q_home[3:7], base.q_home[3:7])  # pelvis orientation unchanged
    np.testing.assert_array_equal(g.q_home[g.waist_qpos_adr], base.q_home[g.waist_qpos_adr])
    for s in SIDES:
        np.testing.assert_array_equal(g.q_home[g.arm_qpos_adr[s]], base.q_home[g.arm_qpos_adr[s]])
        for joint, value in LEG_PITCH_RAD.items():
            lo, hi = g.model.joint(f"{s}_{joint}_joint").range
            assert lo <= value <= hi
        R = g.data.xmat[g.model.body(f"{s}_ankle_roll_link").id].reshape(3, 3)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)  # flat feet


def test_targets_keep_the_unshifted_camera():
    np.testing.assert_allclose(reference_T_world_from_cam(),
                               G1Wuji().T_world_from_torso() @ frames.T_torso_from_cam(), atol=0)
    g = placed_g1()
    moved = g.T_world_from_torso() @ frames.T_torso_from_cam()
    np.testing.assert_allclose(moved[:3, 3] - reference_T_world_from_cam()[:3, 3], PELVIS_OFFSET_M, atol=1e-12)


@pytest.mark.parametrize("placed", [True, False])
def test_render_camera_moves_back_to_the_reference(placed):
    g = placed_g1(placed=placed)
    move_camera_to_reference(g, HEAD_CAMERA)
    d = mujoco.MjData(g.model)
    d.qpos[:] = g.q_home
    mujoco.mj_kinematics(g.model, d)
    mujoco.mj_camlight(g.model, d)  # camera poses are computed here, not in mj_kinematics
    c = g.model.camera(HEAD_CAMERA).id
    T = reference_T_world_from_cam()
    np.testing.assert_allclose(d.cam_xpos[c], T[:3, 3], atol=1e-12)
    # MuJoCo cameras look along -z with +y up; the CV camera looks along +z with +y down.
    np.testing.assert_allclose(d.cam_xmat[c].reshape(3, 3) @ np.diag([1, -1, -1]), T[:3, :3], atol=1e-9)
