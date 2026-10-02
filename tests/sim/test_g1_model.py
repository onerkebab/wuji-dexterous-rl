import mujoco
import numpy as np
import pytest

from wuji_cap.common import frames
from wuji_cap.common import joint_order as jo
from wuji_cap.sim import g1_model

pytestmark = pytest.mark.skipif(not g1_model.g1_xml_path().is_file(), reason="TWIST2 G1 model not found")


@pytest.fixture(scope="module")
def g():
    return g1_model.G1Wuji(64, 48)


def test_composition_counts_and_names(g):
    m = g.model
    assert m.nu == 29 + 40
    assert len(g.frozen_dofs()) == m.nv - 14
    for s in jo.SIDES:
        assert [m.joint(j).name for j in g.arm_jnt_ids[s]] == list(g1_model.arm_joint_names(s))
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_rubber_hand") == -1
        # Wuji SDK-order access works inside the composed model; the hand hangs off the G1 wrist.
        h = g.hands[s]
        assert m.body(h.wrist_body_id).name == jo.side_prefix(s) + "wrist"
        parent = m.body(m.body_parentid[h.wrist_body_id]).name
        assert parent == f"{s}_wrist_yaw_link"


def test_home_pose_is_g1_keyframe(g):
    g1 = mujoco.MjModel.from_xml_path(str(g1_model.g1_xml_path()))
    for j in range(1, g1.njnt):  # skip the free joint
        name = g1.joint(j).name
        jid = mujoco.mj_name2id(g.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert g.q_home[g.model.jnt_qposadr[jid]] == g1.key_qpos[0, g1.jnt_qposadr[j]]
    for s in jo.SIDES:
        assert np.all(g.q_home[g.hands[s].qpos_adr] == 0)


def test_hands_mounted_mirror_symmetric(g):
    """AS-02 uses one mount quaternion for both sides; with mirrored arm poses the hands must mirror."""
    d = mujoco.MjData(g.model)
    d.qpos[:] = g.q_home
    for s in jo.SIDES:
        d.qpos[g.arm_qpos_adr[s]] = 0.0
    mujoco.mj_kinematics(g.model, d)
    M = np.diag([1.0, -1.0, 1.0])
    tips = {s: d.site_xpos[g.hands[s].tip_site_ids] for s in jo.SIDES}
    palms = {s: d.xpos[g.palm_body_ids[s]] for s in jo.SIDES}
    assert np.allclose(palms["left"], M @ palms["right"], atol=1e-4)
    assert np.abs(tips["left"] - tips["right"] @ M).max() < 0.003  # the two MJCFs differ by ~2 mm
    # Fingers point forward (+X), thumbs up (+Z).
    for s in jo.SIDES:
        assert (tips[s][1:, 0] > palms[s][0] + 0.1).all() and tips[s][0, 2] > palms[s][2] + 0.05


def test_head_camera_matches_torso_from_cam(g):
    d = mujoco.MjData(g.model)
    d.qpos[:] = g.q_home
    mujoco.mj_kinematics(g.model, d)
    mujoco.mj_camlight(g.model, d)
    cam = mujoco.mj_name2id(g.model, mujoco.mjtObj.mjOBJ_CAMERA, g1_model.HEAD_CAMERA)
    T_wc = g.T_world_from_torso() @ frames.T_torso_from_cam()
    assert np.allclose(d.cam_xpos[cam], T_wc[:3, 3])
    R_mj = d.cam_xmat[cam].reshape(3, 3)  # MuJoCo camera: x right, y up, looks along -z
    assert np.allclose(R_mj, T_wc[:3, :3] @ np.diag([1.0, -1.0, -1.0]), atol=1e-6)
