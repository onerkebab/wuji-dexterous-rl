"""B4 (W1-B14) tests: usable frames, hold-last playback arrays, and the B2 contract fields on unusable frames."""

import numpy as np
import pytest

from wuji_cap.common import derived_io
from wuji_cap.common import joint_order as jo
from wuji_cap.sim import hold, stub_keypoints
from wuji_cap.sim.retarget import retarget_session

CAM = {"fx": 1000.0, "fy": 1000.0, "cx": 500.0, "cy": 500.0, "width": 1000, "height": 1000}
OUT = {"left": (200, 230), "right": (40, 45)}  # rows pushed out of the image (keypoints stay valid)


def test_knuckles_in_image():
    kp = np.zeros((5, 21, 3))
    kp[..., 2] = 0.5
    kp[1, 5, 0] = 10.0  # index knuckle far right -> out
    kp[2, 8, 0] = 10.0  # index tip out, knuckles in -> seen
    kp[3, :, 2] = -0.5  # behind the camera
    kp[4, 0, 1] = 10.0  # W1-B15: only the wrist past the bottom edge -> seen
    assert hold.knuckles_in_image(kp, CAM).tolist() == [True, False, True, False, True]
    kp[0, 1] = np.nan
    assert hold.knuckles_in_image(kp, None).tolist() == [False, True, True, True, True]


T_S = np.arange(12) / 10.0  # 10 Hz, dt = 0.1 s


def test_playback_holds_exactly_then_returns_at_the_cap():
    """Held rows repeat the previous playback row exactly; after the gap the playback moves toward the tracked pose
    at <= cap * dt per frame along a straight line, then equals the tracked pose."""
    q = np.zeros((12, 2))
    q[6:] = [1.0, -0.5]  # the hand moved 1 rad while unseen
    usable = np.ones(12, bool)
    usable[3:6] = False
    q[~usable] = np.nan
    cmd, held, catching = hold.playback(q, usable, np.zeros(2), T_S, speed_cap=4.0)
    assert np.array_equal(held, ~usable)
    assert (cmd[3:6] == cmd[2]).all() and (cmd[:3] == 0).all()
    assert catching.tolist() == [False] * 6 + [True, True] + [False] * 4  # 0.4 + 0.4 rad, then within 0.4 rad
    step = np.abs(np.diff(cmd, axis=0)).max(1)
    assert step.max() <= 4.0 * 0.1 + 1e-12
    assert np.allclose(cmd[7] / np.linalg.norm(cmd[7]), q[8] / np.linalg.norm(q[8]))  # straight line in joint space
    assert np.array_equal(cmd[8:], q[8:])


def test_playback_starts_from_home_and_is_identity_without_gaps():
    q = np.linspace(0, 1, 12)[:, None] * [0.2, -0.1]
    cmd, held, catching = hold.playback(q, np.ones(12, bool), q[0], T_S, speed_cap=2.0)
    assert np.array_equal(cmd, q) and not held.any() and not catching.any()
    cmd, _, catching = hold.playback(q + 1.0, np.ones(12, bool), q[0], T_S, speed_cap=2.0)  # first pose 1 rad from home
    assert (cmd[0] == q[0]).all() and catching[0] and catching[:6].all() and not catching[-1]


@pytest.fixture(scope="module")
def out_of_view():
    kp, _ = stub_keypoints.make_stub()
    kp = dict(kp)
    for s, (a, b) in OUT.items():
        kp[f"{s}_kp21_cam"] = kp[f"{s}_kp21_cam"].copy()
        kp[f"{s}_kp21_cam"][a:b, :, 0] += 5.0  # 5 m to the side: off the image, still finite and valid
        assert kp[f"{s}_valid"][a:b].all()
    out, _ = retarget_session(kp, cam=CAM)
    return kp, out


@pytest.mark.parametrize("side", jo.SIDES)
def test_unusable_frames_are_nan_and_held(side, out_of_view):
    kp, out = out_of_view
    a, b = OUT[side]
    vis = out[f"{side}_visible"]
    assert not vis[a:b].any() and vis[np.r_[:a, b:len(vis)]][kp[f"{side}_valid"][np.r_[:a, b:len(vis)]]].all()
    v = out[f"{side}_valid"]
    assert not v[a:b].any() and np.isnan(out[f"{side}_q_wuji"][a:b]).all()
    assert np.array_equal(out[f"{side}_held"], ~v)
    cmd, catching = out[f"{side}_q_wuji_cmd"], out[f"{side}_wuji_catching_up"]
    assert (cmd[a:b] == cmd[a - 1]).all()  # hold
    assert v[b] and catching[b]  # the hand moved while unseen: return at the cap, not a jump
    step = np.r_[0.0, np.abs(np.diff(cmd, axis=0)).max(1)]
    dt = np.r_[0.0, np.diff(out["timestamps_s"])]
    assert (step[catching] <= hold.WUJI_SPEED_CAP_RAD_S * dt[catching] + 1e-9).all()  # the cap holds while catching up
    e = b + np.flatnonzero(~catching[b:])[0]  # caught up
    assert np.array_equal(cmd[e], out[f"{side}_q_wuji"][e])
    ok = v & ~catching
    assert np.array_equal(cmd[ok], out[f"{side}_q_wuji"][ok])
    assert derived_io.check_retarget(out) == stub_keypoints.N_FRAMES


def test_contract_check_rejects_arm_on_invalid_frame(out_of_view):
    _, out = out_of_view
    bad = dict(out)
    bad["left_q_g1_arm"] = np.zeros_like(out["left_q_g1_arm"])
    with pytest.raises(ValueError, match="q_g1_arm set on an invalid frame"):
        derived_io.check_retarget(bad)
    bad = dict(out)
    bad["left_q_wuji_cmd"] = out["left_q_wuji_cmd"].copy()
    bad["left_q_wuji_cmd"][0] = np.nan
    with pytest.raises(ValueError, match="q_wuji_cmd"):
        derived_io.check_retarget(bad)
