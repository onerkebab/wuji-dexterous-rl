"""End-to-end edge case: one RGB frame, with one entirely absent hand."""

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

from scripts.mano_overlay import main
from wuji_cap.data.mano_clean import sha256


def test_single_frame_absent_hand_overlay(tmp_path, monkeypatch):
    source = Path("data/raw/ego-clinical-task/unscrew_cap/session_032/metadata.json")
    if not source.exists() or not Path("models/mano_v1_2_clean").exists():
        pytest.skip("Local licensed models required")
    session = tmp_path / "session"
    session.mkdir()
    metadata = json.loads(source.read_text())
    metadata.update(frame_count=1, hands_present=["right"])
    metadata["camera"].update(width=64, height=64, fx=100., fy=100., cx=32., cy=32.)
    (session / "metadata.json").write_text(json.dumps(metadata))
    arrays = {"timestamps_s": np.array([0.])}
    for side, n in (("left", 0), ("right", 1)):
        for name, size in (("hand_pose", 45), ("global_orient", 3), ("transl", 3)):
            arrays[f"{side}_{name}"] = np.zeros((n, size), dtype=np.float32)
        arrays[f"{side}_betas"] = np.zeros(10 if n else 0, dtype=np.float32)
    arrays["right_transl"][0, 2] = .5
    np.savez(session / "mano.npz", **arrays)
    writer = cv2.VideoWriter(str(session / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30., (64, 64))
    assert writer.isOpened()
    writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    validation = {
        "frame_count": 1, "video_decoded_frames": 1,
        "output_sha256": {name: sha256(session / name) for name in ("metadata.json", "mano.npz", "rgb.mp4")},
    }
    (session / "validation.json").write_text(json.dumps(validation))
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["mano_overlay", "--session", str(session), "--output", str(output)])
    main()
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["video"]["output_decoded_frames"] == 1
    assert metrics["hands"]["left"]["valid_frames"] == 0
    assert metrics["hands"]["right"]["valid_frames"] == 1
    assert metrics["timestamps"]["min_step_s"] is None
    assert "left_shape_sign_diagnostic" not in metrics
