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
MuJoCo 3.17 CPU batch; `--jobs` sets parallelism. Missing data, weights and pinned author
inference files are downloaded and hash-checked; results and traces stay outside Git.

[Inference validation](validation/README.md) · [Native parity report](validation/native-parity.json)

## Observations

Ideal proprioception, without robot world-position or contact feedback. [Checkpoint requirements](admission.json) · [Rules](https://github.com/MotionChallenge-Project/MotionChallenge-Core/blob/da0aec4023bbc3d29be67e1fce9fa9ab8a48c4b5/docs/OBSERVATIONS.md).


Initialize root history from the first reference root position instead of measured robot position, which is unavailable under the benchmark observation rules. Then use the author mode 7 reference-forcing inference with the original checkpoint; no robot translation is measured or integrated.
