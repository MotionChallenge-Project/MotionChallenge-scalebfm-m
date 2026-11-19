# Validation

The checks below use the pinned model artifacts and the strict observation contract in [admission.mjwarp.json](admission.mjwarp.json).

| Check | Evidence |
| --- | --- |
| Original network versus converted batch network | [Passed](validation/mjwarp-network-parity.json) |
| Observation construction, history and partial resets | [Passed](validation/tensor-observation-parity.json) |
| Full 4096-slot forward and independent seven-slot comparison | [Passed](validation/full-batch-capacity.json) |
| Official author release and original artifact hashes | [Verified](validation/checkpoint-origin.json) |
| Full LAFAN rollout and independent CPU score verification | [Passed](validation/lafan-full.json) |
| Full SEED rollout and independent CPU score verification | [Passed](validation/seed-full.json) |
| Independent model directory, empty conversion/data cache, one-command evaluation | [Passed](validation/clean-reproduction.json) |

The full-batch check executes four complete 4096-slot forwards, resets slots 1 and 4095, and compares seven selected slots with a separately initialized adapter. Its timing measures adapter inference, not complete simulation throughput.

Full dataset reports bind the runtime, protocol, checkpoint, prepared dataset and result hashes. Scores are independently recomputed on CPU from every saved trace. The empty-cache report records the exact tested identity and the scope of the pipeline check.
