import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from wuji_cap.data.mano_keypoints import export_session, keypoints_from_reconstruction, validate_keypoints
from wuji_cap.data.mano_reconstruct import Reconstruction, make_layer, reconstruct
from wuji_cap.data.mano_session import HandParameters


def synthetic_arrays():
    count = 3
    hand = HandParameters(np.zeros((count, 45)), np.tile([0., 0., np.pi / 2], (count, 1)),
                          np.zeros((count, 3)), np.zeros(10), np.array([True, False, True]))
    result = Reconstruction(np.arange(count * 778 * 3, dtype=float).reshape(count, 778, 3),
                            np.arange(count * 16 * 3, dtype=float).reshape(count, 16, 3), hand.valid.copy())
    return hand, result


def test_mediapipe_order_and_wrist_pose():
    hand, result = synthetic_arrays()
    points, transform, valid = keypoints_from_reconstruction(hand, result)
    # Explicit standard order independent of the production mapping table.
    joint_slots = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    native = [0, 13, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9]
    np.testing.assert_array_equal(points[0, joint_slots], result.joints[0, native])
    np.testing.assert_array_equal(points[0, [4, 8, 12, 16, 20]], result.vertices[0, [744, 320, 443, 554, 671]])
    np.testing.assert_allclose(transform[0, :3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-15)
    np.testing.assert_array_equal(transform[0, :3, 3], result.joints[0, 0])
    assert not np.array_equal(transform[0, :3, 3], hand.transl[0])
    assert np.isnan(points[1]).all() and np.isnan(transform[1]).all()
    np.testing.assert_array_equal(valid, hand.valid)


def test_bad_reconstruction_masks_entire_row():
    hand, result = synthetic_arrays()
    result.vertices[0, 744, 2] = np.inf
    points, transform, valid = keypoints_from_reconstruction(hand, result)
    np.testing.assert_array_equal(valid, [False, False, True])
    assert np.isnan(points[:2]).all() and np.isnan(transform[:2]).all()


@pytest.fixture
def exported(tmp_path):
    """No licensed models needed: exercise the full writer with a fake layer."""
    from unittest.mock import patch
    from wuji_cap.data.mano_clean import sha256
    from wuji_cap.data.mano_session import ManoSession

    hand, result = synthetic_arrays()
    absent = HandParameters(np.full((3, 45), np.nan), np.full((3, 3), np.nan),
                            np.full((3, 3), np.nan), np.full(10, np.nan), np.zeros(3, bool))
    absent_result = Reconstruction(np.full((3, 778, 3), np.nan), np.full((3, 16, 3), np.nan), absent.valid)
    for side in ("LEFT", "RIGHT"):
        (tmp_path / f"MANO_{side}.pkl").write_bytes(side.encode())
    session = ManoSession(tmp_path, {
        "mano": {"model_sha256": {side: sha256(tmp_path / f"MANO_{side.upper()}.pkl") for side in ("left", "right")}},
        "coordinate_frame": "zed_rectified_left_camera", "camera_axes": "+X right, +Y down, +Z forward",
    }, np.array([0., .03, .12]), np.array([0, 1, 2]), {"left": absent, "right": hand}, {"mano.npz": "a" * 64})
    with patch("wuji_cap.data.mano_keypoints.load_session", return_value=session), \
         patch("wuji_cap.data.mano_keypoints.make_layer"), \
         patch("wuji_cap.data.mano_keypoints.reconstruct", side_effect=[absent_result, result]):
        destination = export_session(tmp_path, tmp_path / "derived/session.npz", tmp_path, tmp_path)
    with np.load(destination, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def test_npz_roundtrip_absent_hand_gaps_and_provenance(exported):
    meta = validate_keypoints(exported)
    np.testing.assert_array_equal(exported["frame_index"], [0, 1, 2])
    np.testing.assert_array_equal(exported["timestamps_s"], [0., .03, .12])
    np.testing.assert_array_equal(exported["right_valid"], [True, False, True])
    assert not exported["left_valid"].any()
    for name in ("kp21_cam", "T_cam_from_wrist", "betas"):
        assert np.isnan(exported[f"left_{name}"]).all()
    assert exported["left_kp21_cam"].shape == (3, 21, 3)
    assert exported["meta_json"].dtype.kind == "U"
    assert all({"path", "sha256"} <= item.keys() for item in meta["sources"])
    assert meta["schema_version"] == 2 and len(meta["git_commit"]) == 40
    assert meta["params"]["left_shapedirs_flip"] is False
    assert "AS-A02" in meta["open_assumptions"]


@pytest.mark.parametrize("fault", ["old_key", "dtype", "shape", "invalid_finite", "valid_nan", "rotation", "wrist", "bottom_row", "timestamps", "frame_index", "betas", "meta"])
def test_contract_corruption_rejected(exported, fault):
    arrays = copy.deepcopy(exported)
    if fault == "old_key": arrays["right_wrist_T_cam"] = arrays.pop("right_T_cam_from_wrist")
    if fault == "dtype": arrays["right_valid"] = arrays["right_valid"].astype(int)
    if fault == "shape": arrays["right_kp21_cam"] = arrays["right_kp21_cam"][:, :20]
    if fault == "invalid_finite": arrays["right_kp21_cam"][1] = 0
    if fault == "valid_nan": arrays["right_kp21_cam"][0, 4, 0] = np.nan
    if fault == "rotation": arrays["right_T_cam_from_wrist"][0, :3, :3] = np.diag([1, 1, -1])
    if fault == "wrist": arrays["right_T_cam_from_wrist"][0, :3, 3] = 0
    if fault == "bottom_row": arrays["right_T_cam_from_wrist"][0, 3, 0] = 1
    if fault == "timestamps": arrays["timestamps_s"][2] = .01
    if fault == "frame_index": arrays["frame_index"][2] = 1
    if fault == "betas": arrays["right_betas"][0] = np.nan
    if fault == "meta": arrays["meta_json"] = np.array(json.dumps({"schema_version": 1}))
    with pytest.raises(ValueError):
        validate_keypoints(arrays)


@pytest.mark.parametrize("side", ["left", "right"])
def test_wrist_transform_maps_local_posed_hand_to_camera(side):
    if not Path(f"models/mano_v1_2_clean/MANO_{side.upper()}.pkl").exists():
        pytest.skip("Licensed clean MANO required")
    layer = make_layer("models/mano_v1_2_clean", side)
    hand = HandParameters(np.linspace(-.2, .3, 45)[None], np.array([[.2, -.4, .7]]),
                          np.array([[.1, -.2, .8]]), np.linspace(-.8, .8, 10), np.ones(1, bool))
    camera = reconstruct(hand, layer)
    points, transforms, _ = keypoints_from_reconstruction(hand, camera)
    local_hand = copy.deepcopy(hand)
    local_hand.global_orient[:] = 0
    local_hand.transl[:] = 0
    local = reconstruct(local_hand, layer)
    local_points, _, _ = keypoints_from_reconstruction(local_hand, local)
    centered = local_points[0] - local.joints[0, 0]
    expected = centered @ transforms[0, :3, :3].T + transforms[0, :3, 3]
    np.testing.assert_allclose(points[0], expected, atol=2e-7, rtol=0)
    np.testing.assert_allclose(transforms[0, :3, :3], cv2.Rodrigues(hand.global_orient[0])[0], atol=1e-15)
