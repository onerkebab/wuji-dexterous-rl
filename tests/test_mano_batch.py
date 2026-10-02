import json
from types import SimpleNamespace

import numpy as np
import pytest

from wuji_cap.data import mano_batch as batch


@pytest.fixture
def indexed(tmp_path):
    (tmp_path / "session_index.csv").write_text(
        "session,group,hands,frames,notes\n"
        "session_002,unscrew_cap,right,4,example\n"
        "session_001,unscrew_cap,right,4,example\n"
        "session_003,screw_cap,left,9,excluded\n")
    for name in ("session_001", "session_002"):
        (tmp_path / "unscrew_cap" / name).mkdir(parents=True)
    return tmp_path


def test_inventory_is_complete_sorted_and_activity_scoped(indexed):
    assert [row["session"] for row in batch.discover_sessions(indexed)] == ["session_001", "session_002"]


@pytest.mark.parametrize("fault", ["missing", "extra", "duplicate", "unsafe_name"])
def test_bad_inventory_rejected(indexed, fault):
    if fault == "missing":
        (indexed / "unscrew_cap/session_002").rmdir()
    if fault == "extra":
        (indexed / "unscrew_cap/session_004").mkdir()
    if fault == "duplicate":
        with (indexed / "session_index.csv").open("a") as stream:
            stream.write("session_001,unscrew_cap,right,4,duplicate\n")
    if fault == "unsafe_name":
        path = indexed / "session_index.csv"
        path.write_text(path.read_text().replace("session_001", "../session_001"))
    with pytest.raises(ValueError):
        batch.discover_sessions(indexed)


@pytest.fixture
def statistics_input():
    arrays = {"frame_index": np.arange(4), "timestamps_s": np.array([0., 1 / 30, .1, 2 / 15]),
              "meta_json": np.array(json.dumps({"schema_version": 2, "sources": [], "script": "test",
                  "git_commit": "a" * 40, "git_dirty": False, "created_utc": "test", "params": {}}))}
    hands = {}
    for side in batch.SIDES:
        valid = np.array([True, False, False, True]) if side == "right" else np.zeros(4, bool)
        betas = np.zeros(10) if side == "right" else np.full(10, np.nan)
        points = np.full((4, 21, 3), np.nan)
        points[valid] = [0., 0., 1.]
        transforms = np.full((4, 4, 4), np.nan)
        transforms[valid] = np.eye(4)
        transforms[valid, 2, 3] = 1
        arrays.update({f"{side}_kp21_cam": points, f"{side}_T_cam_from_wrist": transforms,
                       f"{side}_valid": valid, f"{side}_betas": betas})
        hands[side] = SimpleNamespace(valid=valid.copy(), betas=betas.copy())
    # One tip outside the image on the last valid row; still reconstruction-valid.
    arrays["right_kp21_cam"][3, 20, 0] = 10
    session = SimpleNamespace(frame_index=arrays["frame_index"].copy(),
        timestamps_s=arrays["timestamps_s"].copy(), hands=hands,
        metadata={"hands_present": ["right"], "camera": {"fps_nominal": 30., "fx": 100.,
                  "fy": 100., "cx": 50., "cy": 50., "width": 100, "height": 100}})
    return session, arrays


def test_statistics_distinguish_absence_invalidity_bounds_and_time_gaps(statistics_input):
    stats = batch.session_statistics(*statistics_input)
    assert stats["timestamp_span_s"] == pytest.approx(2 / 15)
    spacing = stats["timestamp_spacing"]
    assert spacing["gap_after_rows"] == [1]
    assert spacing["gap_intervals_s"] == pytest.approx([1 / 15])
    assert spacing["uniform_at_nominal_within_1us"] is False
    right, left = stats["hands"]["right"], stats["hands"]["left"]
    assert right["valid_frames"] == 2 and right["valid_fraction"] == .5
    assert right["invalid_row_runs_inclusive"] == [[1, 2]]
    assert right["longest_invalid_run_frames"] == 2
    assert right["all_21_project_inside_image_frames"] == 1
    assert right["per_keypoint_inside_image_frames"] == [2] * 20 + [1]
    assert left["present"] is False and left["valid_frames"] == 0
    assert left["invalid_row_runs_inclusive"] == [[0, 3]]
    json.dumps(stats, allow_nan=False)


def test_single_frame_has_no_spacing_evidence(statistics_input):
    session, arrays = statistics_input
    for key in arrays:
        if key not in ("meta_json", "left_betas", "right_betas"):
            arrays[key] = arrays[key][:1]
    session.frame_index = session.frame_index[:1]
    session.timestamps_s = session.timestamps_s[:1]
    for hand in session.hands.values():
        hand.valid = hand.valid[:1]
    spacing = batch.session_statistics(session, arrays)["timestamp_spacing"]
    assert spacing["interval_count"] == 0 and spacing["gap_count"] == 0
    assert spacing["uniform_at_nominal_within_1us"] is None
    assert spacing["min_s"] is None


@pytest.mark.parametrize("fault", ["timestamps", "frames", "source_invalid"])
def test_export_source_mismatch_rejected(statistics_input, fault):
    session, arrays = statistics_input
    if fault == "timestamps": session.timestamps_s[-1] += .01
    if fault == "frames": session.frame_index[-1] += 1
    if fault == "source_invalid": session.hands["right"].valid[0] = False
    with pytest.raises((AssertionError, ValueError)):
        batch.session_statistics(session, arrays)


@pytest.mark.parametrize("fail", [False, True])
def test_batch_reports_failures_and_continues(indexed, monkeypatch, fail):
    visited = []
    def process(row, *args):
        visited.append(row["session"])
        if fail and row["session"] == "session_001":
            raise ValueError("bad source hash")
        return {"status": "passed", "frame_count": 4}
    monkeypatch.setattr(batch, "process_session", process)
    output = indexed / "results"
    report = batch.run_batch(indexed, indexed / "derived", output, indexed, indexed)
    assert visited == ["session_001", "session_002"]
    assert report["complete"] is (not fail)
    assert report["total_frames_passed"] == (4 if fail else 8)
    if fail:
        assert report["failures"]["session_001"]["message"] == "bad source hash"
    assert json.loads((output / "metrics.json").read_text()) == report


def test_inventory_failure_writes_incomplete_report(indexed):
    (indexed / "unscrew_cap/session_002").rmdir()
    report = batch.run_batch(indexed, indexed / "derived", indexed / "results", indexed, indexed)
    assert not report["complete"] and "inventory" in report["failures"]
    assert not report["sessions"]


def test_stale_archive_is_not_overwritten(indexed, statistics_input, monkeypatch):
    session, arrays = statistics_input
    monkeypatch.setattr(batch, "load_session", lambda path: session)
    target = indexed / "session_001.npz"
    np.savez(target, **arrays)
    original = target.read_bytes()
    def reject(*args):
        raise ValueError("Archive source/model provenance does not match current inputs")
    monkeypatch.setattr(batch, "verify_provenance", reject)
    with pytest.raises(ValueError, match="provenance"):
        batch.process_session({"session": "session_001", "frames": "4", "hands": "right"},
                              indexed, indexed, indexed / "results", indexed, indexed, 64)
    assert target.read_bytes() == original


@pytest.mark.parametrize("fault", [None, "source", "model", "code", "convention"])
def test_provenance_rejects_changed_inputs_and_conventions(tmp_path, statistics_input, fault):
    from pathlib import Path
    session, arrays = statistics_input
    session.path = tmp_path
    session.source_sha256 = {"mano.npz": "a" * 64}
    session.metadata["mano"] = {"model_sha256": {}}
    sources = [{"path": str(tmp_path / "mano.npz"), "sha256": "a" * 64}]
    for side in batch.SIDES:
        path = tmp_path / f"MANO_{side.upper()}.pkl"
        path.write_bytes(side.encode())
        digest = batch.sha256(path)
        session.metadata["mano"]["model_sha256"][side] = digest
        sources.append({"path": str(path), "sha256": digest})
    repo = Path(batch.__file__).resolve().parents[2]
    meta = json.loads(arrays["meta_json"].item())
    meta.update({"sources": sources, "params": {"use_pca": False, "flat_hand_mean": True,
        "left_shapedirs_flip": False, "interpolate": False},
        "code_sha256": {f"wuji_cap/data/{name}": batch.sha256(repo / "wuji_cap/data" / name)
                        for name in ("mano_keypoints.py", "mano_session.py", "mano_reconstruct.py", "mano_clean.py")}})
    if fault == "source": sources[0]["sha256"] = "b" * 64
    if fault == "model": (tmp_path / "MANO_LEFT.pkl").write_bytes(b"changed model")
    if fault == "code": meta["code_sha256"]["wuji_cap/data/mano_keypoints.py"] = "c" * 64
    if fault == "convention": meta["params"]["interpolate"] = True
    arrays["meta_json"] = np.array(json.dumps(meta))
    if fault is None:
        batch.verify_provenance(session, arrays, tmp_path, tmp_path)
    else:
        with pytest.raises(ValueError, match="mismatch|provenance"):
            batch.verify_provenance(session, arrays, tmp_path, tmp_path)


def test_cli_failure_exits_nonzero(monkeypatch):
    monkeypatch.setattr("sys.argv", ["mano_batch"])
    monkeypatch.setattr(batch, "run_batch", lambda *args: {"complete": False, "failures": {"session_001": {}}})
    with pytest.raises(SystemExit) as error:
        batch.main()
    assert error.value.code == 1
