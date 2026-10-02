import numpy as np

from wuji_cap.common import frames
from wuji_cap.sim.scene import ZED_FOVY_DEG, FloatingHands


def test_zed_camera_matches_optical_convention():
    """A point given in the optical camera frame lands on the matching side of the MuJoCo camera image.

    MuJoCo camera frame: x right, y up, looking along -z. Regression: cam_xmat must be computed
    (mj_camlight), not left at zero.
    """
    s = FloatingHands(64, 36)
    s.forward()
    cam = 0
    R, p0 = s.data.cam_xmat[cam].reshape(3, 3), s.data.cam_xpos[cam]
    assert np.allclose(R @ R.T, np.eye(3))
    T_wc = frames.T_world_from_cam()
    for p_opt, expect in [([0.2, 0, 1], (1, 0)), ([0, 0.2, 1], (0, -1))]:
        pc = R.T @ (frames.apply_T(T_wc, np.array(p_opt, float)) - p0)
        assert pc[2] < 0  # in front of the camera
        assert np.sign(pc[0]) == expect[0] or expect[0] == 0
        assert np.sign(pc[1]) == expect[1] or expect[1] == 0
    assert np.isclose(s.model.cam_fovy[cam], ZED_FOVY_DEG) and 35 < ZED_FOVY_DEG < 45


def test_hidden_hand_moves_out_of_view():
    s = FloatingHands(64, 36)
    s.set_hand("left", None, None)
    s.forward()
    assert s.data.xpos[s.hands["left"].wrist_body_id][2] < -5
