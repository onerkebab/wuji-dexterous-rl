import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wuji_cap.common import frames


def test_world_from_cam_axes():
    R = frames.R_WORLD_FROM_CAM
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R @ [0, 0, 1], [1, 0, 0])  # optical forward -> world forward
    assert np.allclose(R @ [0, 1, 0], [0, 0, -1])  # image down -> world down
    assert np.allclose(R @ [1, 0, 0], [0, -1, 0])  # image right -> world right (-Y)


def test_point_above_camera_is_up_in_world():
    T = frames.T_world_from_cam([0.0, 0.0, 1.5])
    p_world = frames.apply_T(T, np.array([0.0, -0.2, 1.0]))  # 1 m ahead, 0.2 m above the optical axis
    assert np.allclose(p_world, [1.0, 0.0, 1.7])


def test_inv_T_roundtrip():
    rng = np.random.default_rng(0)
    T = np.array([frames.make_T(Rotation.random(random_state=i).as_matrix(), rng.normal(size=3)) for i in range(5)])
    assert np.allclose(frames.inv_T(T) @ T, np.eye(4))


def test_quaternion_conventions_match_mujoco():
    R = Rotation.random(random_state=3)
    q_mj = np.zeros(4)
    mujoco.mju_mat2Quat(q_mj, R.as_matrix().ravel())
    q_wxyz = frames.quat_wxyz_from_xyzw(R.as_quat())
    assert np.allclose(q_mj, q_wxyz) or np.allclose(q_mj, -q_wxyz)
    assert np.allclose(frames.quat_xyzw_from_wxyz(q_wxyz), R.as_quat())


@pytest.mark.parametrize("scale", [None, 0.8])
def test_umeyama_recovers_similarity(scale):
    rng = np.random.default_rng(1)
    src = rng.normal(size=(6, 3))
    R = Rotation.random(random_state=2).as_matrix()
    t = rng.normal(size=3)
    dst = 0.8 * src @ R.T + t
    s, R_hat, t_hat = frames.umeyama(src, dst, scale=scale)
    assert np.isclose(s, 0.8) and np.allclose(R_hat, R) and np.allclose(t_hat, t)


def test_umeyama_never_returns_reflection():
    rng = np.random.default_rng(4)
    src = rng.normal(size=(6, 3))
    dst = src * [1, -1, 1]  # mirror image
    _, R, t = frames.umeyama(src, dst, scale=1.0)
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.sqrt(((src @ R.T + t - dst) ** 2).sum(1).mean()) > 0.1


def test_torso_from_cam_known_points():
    """AS-01 mount: the optical axis points forward and 0.83 rad down; image right is the robot's right."""
    T = frames.T_torso_from_cam()
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    origin = frames.D435_XYZ_IN_TORSO
    assert np.allclose(frames.apply_T(T, np.zeros(3)), origin)
    th = frames.D435_RPY_IN_TORSO[1]
    assert np.allclose(frames.apply_T(T, [0, 0, 1.0]), origin + [np.cos(th), 0, -np.sin(th)])  # forward, pitched down
    assert np.allclose(R @ [1, 0, 0], [0, -1, 0])  # image right -> torso -Y (robot's right)
    assert np.allclose(R @ [0, -1, 0], [np.sin(th), 0, np.cos(th)])  # image up -> up, tilted forward


def test_urdf_rpy_matches_scipy():
    rpy = np.array([0.3, -0.7, 1.1])
    assert np.allclose(frames.rot_from_urdf_rpy(rpy), Rotation.from_euler("xyz", rpy).as_matrix())


def test_d435_constants_match_unitree_urdf():
    """Guard the copied AS-01 constants against the TWIST2 URDF, when the checkout is present."""
    import re
    from pathlib import Path

    urdf = Path(__file__).resolve().parents[3] / "TWIST2/assets/g1/g1_29dof_rev_1_0.urdf"
    if not urdf.is_file():
        pytest.skip(f"{urdf} not found")
    m = re.search(r'<joint name="d435_joint".*?<origin xyz="([^"]+)" rpy="([^"]+)"', urdf.read_text(), re.S)
    assert m and re.search(r'<joint name="d435_joint".*?<parent link="torso_link"', urdf.read_text(), re.S)
    assert np.allclose([float(x) for x in m.group(1).split()], frames.D435_XYZ_IN_TORSO)
    assert np.allclose([float(x) for x in m.group(2).split()], frames.D435_RPY_IN_TORSO)


def test_provenance_hashes_code():
    from wuji_cap.common.provenance import provenance

    p = provenance("scripts/b3_arm_ik.py")
    assert {"scripts/b3_arm_ik.py", "wuji_cap/common/frames.py", "wuji_cap/sim/arm_ik.py"} <= set(p["code_sha256"])
    assert all(len(h) == 64 for h in p["code_sha256"].values())
