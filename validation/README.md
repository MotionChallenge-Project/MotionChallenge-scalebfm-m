# Inference validation

Benchmark adaptation: initialize root history from the first reference root instead of measured robot position, then retain author mode7 reference forcing. The reset differs from the original author code because measured robot position is unavailable under the benchmark rules. Weights, mode7 inference and subsequent reference forcing are unchanged.

[Script](../validate_inference.py) · [Author source pins](../native-sources.json) · [macOS report](native-parity.json) · [Linux report](native-parity-linux.json).

160 cases and4 independent resets, with the original checkpoint. Full benchmark results require separate complete-motion verification.
