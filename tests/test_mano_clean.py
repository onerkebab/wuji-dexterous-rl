import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from wuji_cap.data.mano_clean import (
    _ChArray, _ManoUnpickler, _Select, clean_model, load_original,
)


def test_select_evaluates_indices_not_entire_backing_array():
    source = _ChArray()
    source.x = np.arange(24).reshape(2, 3, 4)
    selection = _Select()
    selection.a = source
    selection.idxs = np.arange(24).reshape(2, 3, 4)[..., :2].ravel()
    selection.preferred_shape = (2, 3, 2)
    np.testing.assert_array_equal(selection.evaluate(), source.x[..., :2])


def test_unknown_chumpy_operation_rejected():
    import io
    loader = _ManoUnpickler(io.BytesIO())
    with pytest.raises(ValueError, match="Unsupported chumpy"):
        loader.find_class("chumpy.ch_ops", "Add")


def test_hash_checked_before_unpickling(tmp_path):
    source = tmp_path / "bad.pkl"
    source.write_bytes(b"not a pickle")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_original(source, "0" * 64)


@pytest.mark.parametrize("side", ["left", "right"])
def test_real_model_cleaning_preserves_all_arrays(side, tmp_path):
    source = Path(f"models/mano_v1_2/models/MANO_{side.upper()}.pkl")
    meta = Path("data/raw/ego-clinical-task/unscrew_cap/session_032/metadata.json")
    if not source.exists() or not meta.exists():
        pytest.skip("Licensed MANO assets and session metadata required")
    expected = json.loads(meta.read_text())["mano"]["model_sha256"][side]
    original_bytes = source.read_bytes()
    original = load_original(source, expected)
    # A second, direct check of the actual asset selection (not _plain again).
    with source.open("rb") as stream:
        raw = _ManoUnpickler(stream, encoding="latin1").load()
    backing = raw["shapedirs"].a.x
    indices = np.arange(backing.size).reshape(backing.shape)[..., :10].ravel()
    np.testing.assert_array_equal(raw["shapedirs"].idxs, indices)
    np.testing.assert_array_equal(original["shapedirs"], backing[..., :10])
    destination = tmp_path / source.name
    metrics = clean_model(source, destination, expected)
    with destination.open("rb") as stream:
        cleaned = pickle.load(stream)
    for key in original:
        np.testing.assert_equal(cleaned[key], original[key])
    assert source.read_bytes() == original_bytes
    assert metrics["all_arrays_roundtrip_exact"]
    assert clean_model(source, destination, expected) == metrics
    with pytest.raises(ValueError, match="overwrite the original"):
        clean_model(source, source, expected)
