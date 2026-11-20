# ScaleBFM-M

[Author repository](https://github.com/zengweishuai/ScaleBFM) · [Checkpoint provenance](provenance.json) · [Strict observation contract](admission.mjwarp.json)

## Install

```bash
git clone --recurse-submodules https://github.com/MotionChallenge-Project/MotionChallenge-scalebfm-m.git
cd MotionChallenge-scalebfm-m
uv sync --locked
```

Python 3.12 and an NVIDIA GPU are required. The repository pins Core, Python dependencies, source checkpoints, and auxiliary artifacts. Weights are converted locally to safetensors and executed as one PyTorch batch on the selected GPU.

## Evaluate

Download the original G1 LAFAN or BONES-SEED motions from the author. Pass the motion directory, or the original SEED `g1.tar.gz` archive:

```bash
uv run evaluate.py --dataset lafan --data /path/to/lafan --device cuda:0 --output runs/lafan
uv run evaluate.py --dataset seed --data /path/to/g1.tar.gz --device cuda:0 --output runs/seed
```

The command downloads and checks model artifacts, converts weights, prepares reference poses, evaluates all motions, and independently recomputes scores from saved traces. Core applies the public SEED inclusion list. Users maintain their local motion files.

For local LeRobot v3 motions, add `--data-format lerobot-v3`. For downloaded model files, use `--weights /path/to/original-checkpoint`; auxiliary files use `--artifact NAME=/path/to/file`. `--offline` uses only supplied or cached artifacts. `--cache` selects the working cache.

Batch size is the number of motions, capped at 4096. All active slots execute in one forward call; longest motions are scheduled first and completed slots are refilled. Use `--slots` to select an explicit capacity. The control interval is 20 ms with four 5 ms MuJoCo Warp steps, contact friction 1.0, Newton 50 iterations and 20 line-search iterations.

## Model artifacts

The source is the author's `model_22200.pt` checkpoint. The converter writes the Transformer state dictionary as safetensors and checks it against the pinned native inference wrapper. Five auxiliary artifacts are declared in `runtime.mjwarp.json`: `network`, `export`, `robot`, `metadata` and `mode_table`. The evaluator downloads them automatically; each can also be supplied with `--artifact NAME=/path/to/file`.

[Author checkpoint provenance](validation/checkpoint-origin.json) records the official release, original artifact hashes and conversion scope.

## Observations and verification

The policy receives joint positions and velocities, root orientation and angular velocity. Actual global position and linear velocity are excluded from every observation and history frame. Motion references contain root and joint information; this adapter computes its own reference body features. Core precomputes reference FK before rollout and reads actual body poses from the simulator for scoring.

- [Network conversion parity](validation/mjwarp-network-parity.json)
- [Observation, history and reset parity](validation/tensor-observation-parity.json)
- [Full 4096-slot inference](validation/full-batch-capacity.json)

`validate_tensor.py --prepared PATH --weights CONVERTED --source ORIGINAL --artifact-map ARTIFACTS_JSON --device cuda:0` reproduces the observation check; the artifact map contains `{"artifacts": {"NAME": {"path": "PATH"}}}` for the five entries in `runtime.mjwarp.json`. `validate_full_batch.py --prepared PATH --weights CONVERTED --device cuda:0` uses the prepared motions and converted weights and validates full-batch execution against seven independently initialized slots. The one-command evaluator writes `result.json`, traces and an independent verification report under `--output`.

[Current validation summary](VALIDATION.md) links the conversion, full-batch, complete dataset and independent-cache evidence.

## License

Adapter: [Apache-2.0](LICENSE). [Upstream notices and licenses](THIRD_PARTY_NOTICES.md). Checkpoint and dataset terms are specified by their authors.

BONES-SEED evaluation uses a feasibility-filtered subset of the original G1 motion data. Download the original archive; Core applies one public inclusion list of 129,561 motions shared by every model.

<!-- current-evaluation-evidence -->

## Current evaluation evidence

[LAFAN1 full evaluation](validation/lafan-evaluation.json) · [BONES-SEED full evaluation](validation/seed-evaluation.json)

## Metric

MPJPE (Mean Per-Joint Position Error) measures the 3D Euclidean distance between actual and reference keypoint positions. The protocol defines these keypoints at robot body-frame origins: 14 points for G1. Errors are averaged over keypoints and scoring samples within each motion, then motions are weighted equally. Results report full-motion and first-failure-prefix scores with no alignment, XY alignment, and XY + yaw alignment. XY alignment removes pelvis horizontal translation while retaining height error. Leaderboard values are reported in millimetres (mm).
