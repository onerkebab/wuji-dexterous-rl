import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from wuji_cap.data.mano_clean import load_original
from wuji_cap.data.mano_reconstruct import make_layer, numpy_reference, reconstruct
from wuji_cap.data.mano_session import HandParameters, load_session


@pytest.fixture(params=["left", "right"])
def model_and_session(request):
    side = request.param
    path = Path("data/raw/ego-clinical-task/unscrew_cap/session_032")
    model_path = Path(f"models/mano_v1_2/models/MANO_{side.upper()}.pkl")
    if not path.exists() or not model_path.exists() or not Path("models/mano_v1_2_clean").exists():
        pytest.skip("Licensed original/clean MANO and reference session required")
    session = load_session(path)
    model = load_original(model_path, session.metadata["mano"]["model_sha256"][side])
    return side, model, session


def test_smplx_matches_independent_original_arrays(model_and_session):
    side, model, session = model_and_session
    rows = np.array([0, 73, 177, 353])
    hand = session.hands[side]
    params = HandParameters(hand.hand_pose[rows], hand.global_orient[rows], hand.transl[rows], hand.betas, hand.valid[rows])
    result = reconstruct(params, make_layer("models/mano_v1_2_clean", side), batch_size=3)
    for i in range(len(rows)):
        vertices, joints = numpy_reference(model, params.betas, params.hand_pose[i], params.global_orient[i], params.transl[i])
        np.testing.assert_allclose(result.vertices[i], vertices, rtol=0, atol=2e-7)
        np.testing.assert_allclose(result.joints[i], joints, rtol=0, atol=2e-7)


def test_zero_pose_rotation_translation_and_shape(model_and_session):
    side, model, _ = model_and_session
    betas = np.linspace(-.8, .8, 10)
    angle = np.array([.2, -.3, .4])
    translation = np.array([.1, -.2, .7])
    shaped = model["v_template"] + model["shapedirs"] @ betas
    root = (model["J_regressor"] @ shaped)[0]
    expected = (shaped - root) @ cv2.Rodrigues(angle)[0].T + root + translation
    params = HandParameters(np.zeros((1, 45)), angle[None], translation[None], betas, np.ones(1, bool))
    result = reconstruct(params, make_layer("models/mano_v1_2_clean", side))
    np.testing.assert_allclose(result.vertices[0], expected, rtol=0, atol=2e-7)
    np.testing.assert_allclose(result.joints[0, 0], root + translation, rtol=0, atol=2e-7)
    assert np.linalg.norm(result.joints[0, 0] - translation) > .01


def test_invalid_frames_not_reconstructed(model_and_session):
    side, _, session = model_and_session
    hand = copy.deepcopy(session.hands[side])
    hand.valid[:] = False
    hand.valid[1] = True
    result = reconstruct(hand, make_layer("models/mano_v1_2_clean", side))
    assert np.isnan(result.vertices[~hand.valid]).all()
    assert np.isfinite(result.vertices[1]).all()
    np.testing.assert_array_equal(result.valid, hand.valid)


def test_left_shape_diagnostic_is_explicit_and_does_not_change_asset():
    if not Path("models/mano_v1_2_clean/MANO_LEFT.pkl").exists():
        pytest.skip("Clean models required")
    original = make_layer("models/mano_v1_2_clean", "left")
    flipped = make_layer("models/mano_v1_2_clean", "left", flip_left_shapedirs=True)
    np.testing.assert_array_equal(flipped.shapedirs[:, 0].numpy(), -original.shapedirs[:, 0].numpy())
    np.testing.assert_array_equal(flipped.shapedirs[:, 1:].numpy(), original.shapedirs[:, 1:].numpy())
    reloaded = make_layer("models/mano_v1_2_clean", "left")
    np.testing.assert_array_equal(original.shapedirs.numpy(), reloaded.shapedirs.numpy())



def test_wrong_mean_flag_is_detectably_different(model_and_session):
    import smplx
    side, _, session = model_and_session
    hand = session.hands[side]
    rows = slice(70, 71)
    params = HandParameters(hand.hand_pose[rows], hand.global_orient[rows], hand.transl[rows], hand.betas, hand.valid[rows])
    expected = reconstruct(params, make_layer("models/mano_v1_2_clean", side))
    wrong = smplx.MANO("models/mano_v1_2_clean", is_rhand=side == "right", use_pca=False, flat_hand_mean=False)
    incorrect = reconstruct(params, wrong)
    assert np.max(np.abs(incorrect.vertices - expected.vertices)) > .001
