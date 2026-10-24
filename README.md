# ScaleBFM M

[Author project](https://scalebfm.github.io/) · [Checkpoint](https://huggingface.co/WeishuaiZeng/ScaleBFM/tree/2eec74e0f326076a706702654168efc489ef604a/checkpoint/humanoid_transformer_m) · [Provenance](provenance.json)

## Install and run

```bash
git clone --recurse-submodules https://github.com/MotionChallenge-Project/MotionChallenge-scalebfm-m.git
cd MotionChallenge-scalebfm-m
uv sync
uv run evaluate.py
```

Produces success, completion and six MPBPE values in a verified `result.json`.
MuJoCo3.17 CPU batch; `--jobs 8` sets parallelism. Missing data, weights and pinned author
inference files are downloaded and hash-checked; results and traces stay outside Git.

[Inference validation](validation/README.md) · [Native parity report](validation/native-parity.json)

## Observations

Ideal proprioception, without robot world-position or contact feedback. [Checkpoint requirements](admission.json) · [Rules](MotionChallenge-Core/docs/OBSERVATIONS.md).

Evaluation is unavailable until this checkpoint has a verified author-supported
inference path under these rules. The admission record describes the gap.
