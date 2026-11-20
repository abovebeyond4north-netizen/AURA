# v72 — Door route recovery and immediate-reset stress

Section 08 remains partial. v70 is the comparator, v71 is an experimental rejected candidate, and v72 is opt-in. A reproduced v71-era failure involved a previously observed open door being excluded from paths solely by its historical landmark. The new controller temporarily excludes this landmark from the blocked set only while recovering to the observed empty doorway. Closed/unknown doors remain blocked; no map tile is invented and pose commitment requirements are unchanged. The combined candidate inherits v71 initial calibration and v72 route recovery.

Delayed resets: development seed 98178, baseline 18/24 vs recovery 24/24; fresh seed 98201, baseline 20/24 vs recovery 24/24. Combined 38/48 vs 48/48; no regressions or audited pose errors. Failure-adjusted actions 1558 vs 1177 (failures charged 65). Immediate resets: development seed 98202, baseline 16/24, v71 20/24, recovery 20/24, combined 24/24. Fresh seed 98203: respectively 12/24, 15/24, 16/24, 21/24; combined has two audited initial mapping errors and zero pose errors. Fresh combined failures: two no-isolated-approach failures and one unavailable return route. These are controlled public seeds and not an independent holdout. No criterion upgraded, no default activation or promotion. Map-based recovery is not general exploration through unknown cells.

All five data files retain failures and action sequences (including duplicated development comparisons, not independent trials). Two fresh datasets exactly replayed (144 records). Focused suite 20 passed. Non-verifier partition printed 222 passed, five skipped, five subtests passed; outer command timed out and no shell exit marker was captured. The protected verifier partition was not rerun in v72; previous v67 separate verifier evidence is retained. Installed Gymnasium/TensorBoard absent (Python 3.13.5). The v71 source snapshot has no Git metadata or bundle; past evidence is carried forward but ancestry cannot be independently re-established from this archive. Original standalone master prompt was not included; acceptance register used as the comparison. No protected verifier files changed. 13/31 locally verified, 14 partial, 3 open, 1 ongoing, unchanged.

Reproduce from archive root (set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1):
```
python -m experiments.door_route_recovery_study --seed 98178 --output /tmp/v72_dev.json
python -m experiments.door_route_recovery_study --seed 98201 --output /tmp/v72_fresh.json
python -m experiments.door_route_recovery_study --seed 98202 --reset-start 0 --output /tmp/v72_immediate.json
python -m experiments.door_route_recovery_study --seed 98202 --reset-start 0 --combined --output /tmp/v72_combo_dev.json
python -m experiments.door_route_recovery_study --seed 98203 --reset-start 0 --combined --output /tmp/v72_combo_fresh.json
python -m pytest -o addopts='' -q tests/test_door_route_recovery.py tests/test_probabilistic_calibration.py tests/test_continuous_hazard.py tests/test_hazard_dynamics.py tests/test_adaptive_hazard.py
python -m pytest -o addopts='' -q --ignore=tests/test_verifier.py
```
Next: recover from no isolated approach without weakening causal intervention constraints; verify operational mapping predictions separately from certified facts; improve stochastic dynamics learning; recover official runtime and externally administered tests.
