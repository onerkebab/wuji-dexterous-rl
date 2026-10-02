# Wuji dexterous manipulation

Kinematic retargeting of egocentric cap-unscrewing demonstrations to Wuji Beta 2 hands on a Unitree G1. This is the data and simulation foundation for future reinforcement learning and assisted teleoperation work.

## Implemented

- Load compact MANO recordings, verify model hashes, and convert the original models to chumpy-free arrays.
- Reconstruct both hands and export 21 keypoints per hand in MediaPipe order, with original timestamps, validity masks, and source provenance.
- Retarget hand motion to 20 Wuji joints per hand in SDK order.
- Compose the G1 and Wuji models in MuJoCo and solve inverse kinematics for the two seven-joint arms.
- Hold the last usable pose when knuckles leave the image, with limited-speed returns after gaps.
- Search for a constant robot placement while keeping demonstration targets fixed in the world.
- Render skeleton overlays, floating-hand replays, RGB/hand/robot triptychs, and placement comparisons.

The current pipeline is kinematic. Contact dynamics, cap manipulation policies, RL training, and hardware execution are future work.

## Recorded results

The four evaluated sessions contain 2,438 frames. The integrated robot placement reduced reach failures from 353 to 6 hand-frames across the batch. A reach failure means position error above 10 mm or orientation error above 5 degrees on a usable frame.

| Session | Frames | Left reach failures | Right reach failures |
| --- | ---: | ---: | ---: |
| session_016 | 675 | 0 | 0 |
| session_017 | 625 | 2 | 3 |
| session_021 | 784 | 0 | 1 |
| session_032 | 354 | 0 | 0 |

Evidence is generated locally in `results/20261001-b4-v2-batch/summary.json` and `results/20261001-b5-batch/summary.json`, with per-session `b3/metrics.json` files and plots. Generated results, datasets, and model files are excluded from Git, apart from one regression-test fixture; the commands below regenerate outputs with a new experiment ID.

Per-frame arm IK can still switch between solution branches and produce abrupt motion. Limiting every playback step caused substantial tracking lag and was not retained; trajectory-level IK remains open. The speed limits apply to returns from gaps, not all tracking frames. MANO image-fit discrepancies and approximate wrist mounts also remain limitations.

## Setup

Use Python 3.12 or newer and `uv`:

```bash
uv sync --locked
```

The lockfile includes CPU PyTorch, MuJoCo, mink, smplx, and `wuji-retargeting` pinned to `531f6ed4250b475d2e9231f54e988fc9b1c5b4ea`. Isaac Sim, Isaac Lab, and the TWIST2 Python environment are not needed for this pipeline.

Provide these external assets separately:

| Asset | Expected location | Tested revision |
| --- | --- | --- |
| Wuji description | `../wuji-description` | `c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95` |
| TWIST2 G1 assets | `../TWIST2/assets/g1/g1_sim2sim_29dof.xml` and referenced meshes | TWIST2 `cf391dd7ae9aa2f1fb8fce3541d483121fc83ed1` |
| MANO v1.2 originals | `models/mano_v1_2/models/MANO_LEFT.pkl` and `MANO_RIGHT.pkl` | hashes verified against session metadata |
| Demonstrations | `data/raw/ego-clinical-task/unscrew_cap/` | `LIDAR-GT/ego-clinical-task` |

`WUJI_DESCRIPTION_DIR` and `TWIST2_DIR` override the robot asset roots. Robot repositories: [Wuji description](https://github.com/wuji-technology/wuji-description) and [TWIST2 fork](https://github.com/MLeggiero/TWIST2). Obtain MANO models separately under their license. Dataset access may require Hugging Face authentication.

```bash
uv run hf download LIDAR-GT/ego-clinical-task --repo-type dataset \
  --include 'unscrew_cap/*' 'session_index.csv' \
  --local-dir data/raw/ego-clinical-task
```

## Run

From the repository root, with the assets above available:

```bash
# Verify and convert MANO models; reconstruct and export all four sessions.
uv run python -m scripts.mano_clean
uv run python -m wuji_cap.data.mano_batch

# Retarget fingers, solve arm IK, and write batch metrics.
MUJOCO_GL=egl uv run python scripts/b3_batch.py --exp-id reproduction-batch

# Render RGB, floating hands, and the placed G1 side by side.
MUJOCO_GL=egl uv run python scripts/b3_triptych.py \
  --session session_032 --exp-id reproduction-triptych

# Run tests; asset-dependent tests require the external files above.
MUJOCO_GL=egl uv run pytest -q
```

Keypoints are written to `data/derived/keypoints/`; robot joint trajectories go to `data/derived/retarget/`. Invalid observations remain NaN in the measurement arrays. Separate `*_cmd` arrays provide held and returning poses for playback. Preserve the recorded timestamps: some sessions contain sampling gaps.

## Layout

```text
wuji_cap/data/     MANO loading, reconstruction, and keypoint export
wuji_cap/common/   coordinate frames, joint ordering, and provenance
wuji_cap/sim/      robot models, retargeting, placement, IK, and rendering
scripts/          experiment and visualization entry points
tests/            reconstruction, joint mapping, IK, and playback checks
```

## License and attribution

Project code is available under the [MIT License](LICENSE). The G1/Wuji model composition in `wuji_cap/sim/g1_model.py` is adapted from TWIST2's `deploy_real/wuji_mujoco_model.py`; its MIT notice is retained in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). External robot assets, MANO models, datasets, and dependencies retain their own licenses.
