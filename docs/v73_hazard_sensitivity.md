# v73 — Hazard sensitivity and fresh replication

The v72 combined controller is compared against otherwise identical subclasses with fixed hazard .08 and .15 (baseline .03). Immediate resets, same public matched layout/action/noise budget, seeds 98203 (previously exposed) and 98204 (fresh). Each controller reached 21/24 on development with two mapping errors and zero audited pose errors; all reached 24/24 fresh with zero recorded mapping or pose errors. Failure-adjusted costs development 672 each; fresh 494,498,494 respectively. Raising hazard does not resolve the known mapping errors and is not promoted. Operational mapping audit inherited from v72 samples after key calibration, not every action; it is not a certification. All 144 records preserved, 72 fresh records replayed byte-identically. Focused existing tests 11 passed. No status upgrade. 636 prior manifested files checked. Gymnasium and TensorBoard absent; historical bundle not present in local snapshot, ancestry not independently rerun. No full suite or independent evaluation this run. Register remains 13/31 locally verified. Next prioritize no-isolated-approach cases and stochastic action effects rather than fixed hazard tuning.

Reproduce from archive root with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1:

```
python -m experiments.hazard_sensitivity_study --seed 98203 --output /tmp/v73-dev.json
python -m experiments.hazard_sensitivity_study --seed 98204 --output /tmp/v73-fresh.json
python -m pytest -o addopts='' -q tests/test_door_route_recovery.py tests/test_probabilistic_calibration.py tests/test_continuous_hazard.py
```
