"""Convert trusted, hash-verified MANO pickles without importing chumpy.

Only the Ch array and Select operations present in the approved MANO assets
are supported. This is not a general-purpose or safe unpickler for untrusted data.
"""

import hashlib
import io
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse


EXPECTED_SHAPES = {
    "v_template": (778, 3), "f": (1538, 3), "J_regressor": (16, 778),
    "shapedirs": (778, 3, 10), "posedirs": (778, 3, 135),
    "weights": (778, 16), "hands_components": (45, 45),
    "hands_mean": (45,), "kintree_table": (2, 16),
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class _ChArray:
    def __setstate__(self, state):
        self.__dict__.update(state)

    def evaluate(self):
        return np.asarray(self.x)


class _Select(_ChArray):
    def evaluate(self):
        source = _plain(self.a)
        return source.reshape(-1)[self.idxs].reshape(self.preferred_shape)


class _ManoUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("chumpy.ch", "Ch"):
            return _ChArray
        if (module, name) == ("chumpy.reordering", "Select"):
            return _Select
        if module.startswith("chumpy"):
            raise ValueError(f"Unsupported chumpy operation: {module}.{name}")
        if (module, name) == ("scipy.sparse.csc", "csc_matrix"):
            return sparse.csc_matrix
        return super().find_class(module, name)


def _plain(value):
    if isinstance(value, _ChArray):
        return value.evaluate().copy()
    if sparse.issparse(value):
        return value.toarray()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("Object arrays are not supported")
        return value.copy()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(item) for item in value)
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    raise ValueError(f"Unsupported model value: {type(value)}")


def validate_model(model):
    for key, shape in EXPECTED_SHAPES.items():
        value = model[key]
        if not isinstance(value, np.ndarray) or value.shape != shape:
            raise ValueError(f"{key}: expected array {shape}, got {getattr(value, 'shape', None)}")
        if not np.isfinite(value).all():
            raise ValueError(f"{key}: nonfinite values")
    if not np.issubdtype(model["f"].dtype, np.integer):
        raise ValueError("Faces must have integer indices")
    if model["f"].min() < 0 or model["f"].max() >= 778:
        raise ValueError("Face index out of range")


def load_original(path, expected_sha256):
    # Verify the exact bytes that will be unpickled, before executing the pickle.
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"Original MANO hash mismatch: {path}")
    model = _plain(_ManoUnpickler(io.BytesIO(payload), encoding="latin1").load())
    validate_model(model)
    return model


def clean_model(source, destination, expected_sha256):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Cannot overwrite the original model")
    model = load_original(source, expected_sha256)
    payload = pickle.dumps(model, protocol=4)
    restored = pickle.loads(payload)
    validate_model(restored)
    for key, value in model.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, restored[key], strict=True)
        elif value != restored[key]:
            raise ValueError(f"Round-trip mismatch: {key}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Never silently replace an existing different cleaned asset.
    if destination.exists() and destination.read_bytes() != payload:
        raise ValueError(f"Different cleaned model already exists: {destination}")
    destination.write_bytes(payload)
    return {
        "original_sha256": expected_sha256, "clean_sha256": sha256(destination),
        "array_shapes": {key: list(model[key].shape) for key in EXPECTED_SHAPES},
        "all_arrays_roundtrip_exact": True,
    }
