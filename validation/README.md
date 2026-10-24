# Author inference comparison

The validator compares original ScaleBridge reset, history, reference forcing, observation and action processing on160 matched inputs and4 resets. It loads the original raw M weights into the unmodified author CPU network/export wrapper, and independently compares reference FK using the author float32 operation order. Native source bytes are pinned in native-sources.json.

```bash
uv run validate_inference.py --source /path/to/ScaleBFM --weights /path/to/model_22200.pt --metadata /path/to/model_22200_tensorrt_metadata.json --mode-table /path/to/mode_table.pt --output /tmp/scalebfm-native.json
```

This validates CPU deployment semantics; TensorRT/CUDA numeric equivalence is not claimed. Full motion rollout and repeatability are separate checks. The default control mode and reference-forcing behavior are documented in provenance.json.

An initial reference implementation used double-precision MuJoCo FK and exceeded the fixed2e-5 comparison limit on Linux (action2.48e-5). `diagnostics/initial-mujoco-fk-linux.json` preserves that result. Reference features now use the author float32 quaternion operation order; tolerances and benchmark rules are unchanged.

Final CPU comparisons: `native-parity.json` (macOS) and `native-parity-linux.json` (Linux), both zero measured difference.
