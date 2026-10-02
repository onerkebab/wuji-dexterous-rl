"""Export and audit every indexed unscrew_cap session (contract v2).

uv run python -B -m wuji_cap.data.mano_batch
Existing archives are checked and reused; failures never silently overwrite them.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import torch

from .mano_clean import sha256
from .mano_keypoints import FIT_NOTE, SIDES, VALIDITY_NOTE, export_session, validate_keypoints
from .mano_keypoints_review import audit
from .mano_session import load_session, project_camera


def discover_sessions(root):
    """Require the index and local activity directory to agree exactly."""
    root = Path(root)
    with (root / "session_index.csv").open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["group"] == "unscrew_cap"]
    names = [row["session"] for row in rows]
    if not names or len(names) != len(set(names)):
        raise ValueError("Empty or duplicate unscrew_cap index entries")
    for row in rows:
        if not re.fullmatch(r"session_\d+", row["session"]) or int(row["frames"]) <= 0:
            raise ValueError("Invalid session name or frame count in index")
        hands = row["hands"].split("|") if row["hands"] else []
        if len(hands) != len(set(hands)) or not set(hands) <= set(SIDES):
            raise ValueError("Invalid hands in index")
    local = {path.name for path in (root / "unscrew_cap").iterdir() if path.is_dir()}
    if local != set(names):
        raise ValueError(f"Session inventory mismatch: missing={sorted(set(names) - local)}, "
                         f"unindexed={sorted(local - set(names))}")
    return sorted(rows, key=lambda row: row["session"])


def invalid_runs(valid):
    """Inclusive row ranges; whole-session absence is a single invalid run."""
    edges = np.diff(np.r_[False, ~valid, False].astype(int))
    return [[int(start), int(stop - 1)] for start, stop in
            zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def session_statistics(session, arrays):
    validate_keypoints(arrays)
    np.testing.assert_array_equal(arrays["frame_index"], session.frame_index)
    np.testing.assert_array_equal(arrays["timestamps_s"], session.timestamps_s)
    count = len(session.frame_index)
    camera = session.metadata["camera"]
    dt = np.diff(session.timestamps_s)
    nominal = 1 / camera["fps_nominal"]
    gap_rows = np.flatnonzero(dt > 1.5 * nominal)
    stats = {
        "frame_count": count, "hands_present": session.metadata["hands_present"],
        "timestamp_span_s": float(session.timestamps_s[-1] - session.timestamps_s[0]),
        "nominal_video_duration_s": count / camera["fps_nominal"],
        "frame_index_preserved": True, "timestamps_preserved": True,
        "timestamp_spacing": {
            "nominal_fps": camera["fps_nominal"], "interval_count": len(dt),
            "min_s": float(dt.min()) if len(dt) else None,
            "median_s": float(np.median(dt)) if len(dt) else None,
            "max_s": float(dt.max()) if len(dt) else None,
            "max_abs_deviation_from_nominal_s": float(np.abs(dt - nominal).max()) if len(dt) else None,
            "uniform_at_nominal_within_1us": bool(np.all(np.abs(dt - nominal) <= 1e-6)) if len(dt) else None,
            "gap_threshold_s": 1.5 * nominal, "gap_count": len(gap_rows),
            "gap_after_rows": gap_rows.tolist(), "gap_intervals_s": dt[gap_rows].tolist(),
            "note": "Gap means dt > 1.5 / nominal_fps; not a count of missing RGB rows. No interpolation.",
        },
        "hands": {},
    }
    for side in SIDES:
        hand = session.hands[side]
        valid = arrays[f"{side}_valid"]
        np.testing.assert_array_equal(arrays[f"{side}_betas"], hand.betas)
        if np.any(valid & ~hand.valid):
            raise ValueError(f"{side}: source-invalid row marked valid")
        points = arrays[f"{side}_kp21_cam"]
        pixels = project_camera(points, camera)
        inside = (np.isfinite(pixels).all(axis=-1)
                  & (pixels[..., 0] >= 0) & (pixels[..., 0] < camera["width"])
                  & (pixels[..., 1] >= 0) & (pixels[..., 1] < camera["height"]))
        runs = invalid_runs(valid)
        stats["hands"][side] = {
            "present": side in session.metadata["hands_present"],
            "valid_frames": int(valid.sum()), "invalid_frames": int((~valid).sum()),
            "valid_fraction": float(valid.mean()), "source_valid_frames": int(hand.valid.sum()),
            "reconstruction_rejected_frames": int((hand.valid & ~valid).sum()),
            "invalid_row_runs_inclusive": runs,
            "longest_invalid_run_frames": max((b - a + 1 for a, b in runs), default=0),
            "all_21_project_inside_image_frames": int(inside.all(axis=1).sum()),
            "per_keypoint_inside_image_frames": inside.sum(axis=0).tolist(),
            "nonpositive_depth_points_on_valid_rows": int((points[valid, :, 2] <= 0).sum()),
            "projection_note": "Image bounds only; not visibility, occlusion or anatomical fit accuracy.",
        }
    return stats


def verify_provenance(session, arrays, models, originals):
    """Reject stale inputs, models, exporter code or reconstruction conventions."""
    meta = validate_keypoints(arrays)
    sources = {str(Path(item["path"]).resolve()): item["sha256"] for item in meta["sources"]}
    expected = {str((session.path / name).resolve()): digest
                for name, digest in session.source_sha256.items()}
    for side in SIDES:
        name = f"MANO_{side.upper()}.pkl"
        original_hash = sha256(Path(originals) / name)
        if original_hash != session.metadata["mano"]["model_sha256"][side]:
            raise ValueError(f"Original MANO hash mismatch: {side}")
        expected[str((Path(originals) / name).resolve())] = original_hash
        expected[str((Path(models) / name).resolve())] = sha256(Path(models) / name)
    if any(sources.get(path) != digest for path, digest in expected.items()):
        raise ValueError("Archive source/model provenance does not match current inputs")
    repo = Path(__file__).resolve().parents[2]
    for name in ("mano_keypoints.py", "mano_session.py", "mano_reconstruct.py", "mano_clean.py"):
        relative = "wuji_cap/data/" + name
        if meta.get("code_sha256", {}).get(relative) != sha256(repo / relative):
            raise ValueError(f"Archive exporter code mismatch: {name}")
    for key, value in {"use_pca": False, "flat_hand_mean": True,
                       "left_shapedirs_flip": False, "interpolate": False}.items():
        if meta["params"].get(key) is not value:
            raise ValueError(f"Archive reconstruction convention mismatch: {key}")


def process_session(row, root, destination, output, models, originals, batch_size):
    name = row["session"]
    path = root / "unscrew_cap" / name
    session = load_session(path)
    if (len(session.frame_index) != int(row["frames"])
            or set(session.metadata["hands_present"]) != set(filter(None, row["hands"].split("|")))):
        raise ValueError("Session metadata disagrees with index frames/hands")
    target = destination / f"{name}.npz"
    reused = target.exists()
    if not reused:
        export_session(path, target, models, originals, batch_size)
    with np.load(target, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    verify_provenance(session, arrays, models, originals)
    stats = session_statistics(session, arrays)
    review_dir = output / name
    audit(path, target, review_dir, originals)
    review_path = review_dir / "metrics.json"
    review = json.loads(review_path.read_text())
    stats.update({"status": "passed", "reused_existing_archive": reused,
                  "keypoints": {"path": str(target), "sha256": sha256(target)},
                  "audit": {"path": str(review_path), "sha256": sha256(review_path)},
                  "rgb_decoded_frames": review["rgb_decoded_frames"],
                  "reference_checks": {side: {
                      key: review["hands"][side][key] for key in
                      ("sample_frames", "kp21_reference_max_abs_error_m", "wrist_reference_max_abs_error_m")
                  } for side in SIDES}})
    return stats


def run_batch(root, destination, output, models, originals, batch_size=64):
    root, destination, output, models, originals = map(Path, (root, destination, output, models, originals))
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    report = {
        "schema_version": "mano_a3_batch_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "wuji_cap/data/mano_batch.py", "code_sha256": sha256(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()),
        "params": {"root": str(root), "destination": str(destination), "output": str(output),
                   "models": str(models), "originals": str(originals), "batch_size": batch_size},
        "kinematic_only": True, "validity": VALIDITY_NOTE, "fit_caveat": FIT_NOTE,
        "expected_sessions": [], "sessions": {}, "failures": {}, "complete": False,
    }
    try:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        rows = discover_sessions(root)
        report["index"] = {"path": str(root / "session_index.csv"), "sha256": sha256(root / "session_index.csv")}
        report["expected_sessions"] = [row["session"] for row in rows]
        for row in rows:
            name = row["session"]
            print(f"Processing {name}", flush=True)
            try:
                report["sessions"][name] = process_session(row, root, destination, output, models, originals, batch_size)
            except Exception as error:
                report["failures"][name] = {"type": type(error).__name__, "message": str(error)}
    except Exception as error:
        report["failures"]["inventory"] = {"type": type(error).__name__, "message": str(error)}
    report["complete"] = bool(report["expected_sessions"]) and not report["failures"]
    report["total_frames_passed"] = sum(item["frame_count"] for item in report["sessions"].values())
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/ego-clinical-task"))
    parser.add_argument("--destination", type=Path, default=Path("data/derived/keypoints"))
    parser.add_argument("--output", type=Path, default=Path("results/20260930-mano-a3"))
    parser.add_argument("--models", type=Path, default=Path("models/mano_v1_2_clean"))
    parser.add_argument("--originals", type=Path, default=Path("models/mano_v1_2/models"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = run_batch(args.root, args.destination, args.output, args.models, args.originals, args.batch_size)
    print(json.dumps({"complete": report["complete"], "failures": report["failures"]}, indent=2))
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
