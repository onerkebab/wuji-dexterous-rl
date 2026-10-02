import json
from pathlib import Path

import numpy as np
import pytest

from wuji_cap.data.mano_session import load_session, project_camera


@pytest.fixture
def session(tmp_path):
    metadata = {
        "format": "compact_rgb_mano_v3", "frame_count": 3,
        "coordinate_frame": "zed_rectified_left_camera",
        "camera_axes": "+X right, +Y down, +Z forward", "translation_unit": "meter",
        "mano": {"use_pca": False, "flat_hand_mean": True},
        "hands_present": ["right"],
        "camera": {"fx": 100., "fy": 200., "cx": 50., "cy": 60.,
                   "width": 640, "height": 480, "fps_nominal": 30., "dist_coeffs": [0.] * 5},
    }
    arrays = {"timestamps_s": np.array([0., .03, .12])}
    for side, n in (("left", 0), ("right", 3)):
        for name, size in (("hand_pose", 45), ("global_orient", 3), ("transl", 3)):
            arrays[f"{side}_{name}"] = np.zeros((n, size), dtype=np.float32)
        arrays[f"{side}_betas"] = np.zeros(10 if n else 0, dtype=np.float32)
    return tmp_path, metadata, arrays


def write_session(fixture):
    path, metadata, arrays = fixture
    (path / "metadata.json").write_text(json.dumps(metadata))
    np.savez(path / "mano.npz", **arrays)
    return path


def test_absent_hand_and_time_gap_preserved(session):
    result = load_session(write_session(session), verify_hashes=False)
    left = result.hands["left"]
    assert left.hand_pose.shape == (3, 45)
    assert left.betas.shape == (10,)
    assert np.isnan(left.hand_pose).all() and np.isnan(left.betas).all()
    assert not left.valid.any()
    np.testing.assert_array_equal(result.frame_index, [0, 1, 2])
    np.testing.assert_array_equal(result.timestamps_s, [0., .03, .12])
    assert result.hands["right"].valid.all()


def test_nonfinite_row_retained_and_masked(session):
    session[2]["right_hand_pose"][1, 2] = np.nan
    result = load_session(write_session(session), verify_hashes=False)
    right = result.hands["right"]
    np.testing.assert_array_equal(right.valid, [True, False, True])
    assert np.isnan(right.transl[1]).all()
    assert np.isnan(right.global_orient[1]).all()


@pytest.mark.parametrize("fault", ["pca", "mean", "unit", "axes", "timestamp", "count", "shape", "betas", "presence", "distortion"])
def test_invalid_input_rejected(session, fault):
    _, meta, arrays = session
    if fault == "pca": meta["mano"]["use_pca"] = True
    if fault == "mean": meta["mano"]["flat_hand_mean"] = False
    if fault == "unit": meta["translation_unit"] = "mm"
    if fault == "axes": meta["camera_axes"] = "+Y up"
    if fault == "timestamp": arrays["timestamps_s"][2] = .02
    if fault == "count": meta["frame_count"] = 2
    if fault == "shape": arrays["right_hand_pose"] = np.zeros((3, 15, 3))
    if fault == "betas": arrays["right_betas"][0] = np.nan
    if fault == "presence": meta["hands_present"] = ["left", "right"]
    if fault == "distortion": meta["camera"]["dist_coeffs"][0] = .2
    with pytest.raises(ValueError):
        load_session(write_session(session), verify_hashes=False)


def test_hash_mismatch_rejected(session):
    path = write_session(session)
    (path / "validation.json").write_text(json.dumps({
        "frame_count": 3, "video_decoded_frames": 3,
        "output_sha256": {name: "0" * 64 for name in ("metadata.json", "mano.npz", "rgb.mp4")},
    }))
    with pytest.raises(ValueError, match="Session hash mismatch"):
        load_session(path)


def test_camera_projection_and_nonpositive_depth():
    camera = {"fx": 100., "fy": 200., "cx": 50., "cy": 60.}
    points = np.array([[1., 2., 2.], [0., 0., 1.], [1., 1., 0.], [1., 1., -1.]])
    projected = project_camera(points, camera)
    np.testing.assert_allclose(projected[:2], [[100., 260.], [50., 60.]])
    assert np.isnan(projected[2:]).all()
    np.testing.assert_allclose(project_camera(points[None], camera)[0], projected, equal_nan=True)


def test_reference_session_integrity():
    path = Path("data/raw/ego-clinical-task/unscrew_cap/session_032")
    if not path.exists(): pytest.skip("Local session required")
    result = load_session(path)
    assert len(result.timestamps_s) == 354
    assert all(hand.valid.all() for hand in result.hands.values())
