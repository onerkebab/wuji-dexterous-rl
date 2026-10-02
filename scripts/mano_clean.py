"""Run with: uv run python -m scripts.mano_clean."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import smplx
import torch

from wuji_cap.data.mano_clean import clean_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/raw/ego-clinical-task/unscrew_cap/session_032/metadata.json"))
    parser.add_argument("--source", type=Path, default=Path("models/mano_v1_2/models"))
    parser.add_argument("--destination", type=Path, default=Path("models/mano_v1_2_clean"))
    parser.add_argument("--output", type=Path, default=Path("results/20260929-mano-a0/metrics.json"))
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    metrics = {"created_utc": datetime.now(timezone.utc).isoformat(), "hands": {}}
    for side in ("left", "right"):
        name = f"MANO_{side.upper()}.pkl"
        checks = clean_model(args.source / name, args.destination / name,
                             metadata["mano"]["model_sha256"][side])
        layer = smplx.MANO(str(args.destination), is_rhand=side == "right",
                           use_pca=False, flat_hand_mean=True)
        with torch.no_grad():
            result = layer()
        assert result.vertices.shape == (1, 778, 3)
        assert result.joints.shape == (1, 16, 3)
        assert np.isfinite(result.vertices.numpy()).all()
        checks["smplx_zero_pose_finite"] = True
        metrics["hands"][side] = checks
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
