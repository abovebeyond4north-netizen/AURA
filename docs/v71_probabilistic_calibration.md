# v71 probabilistic initial calibration — rejected candidate

Candidate propagates mapping resets during initial probes, requires unique pose and 12 map matches, and uses posterior MAP only at probability >= .8. Public matched seeds 98177/98178: baseline v70 40/48, candidate 38/48; one improved, three regressed (paired exact p=.625). Failure-adjusted cost 1426 vs 1461. Both have zero audited pose errors; legacy mapping audit is limited and not certification of operational guesses. Neither has initial calibration failures on these seeds, so the intended benefit is unestablished. All failures are return-route failures. Candidate remains opt-in, not promoted. 48 records byte-identically replayed; 13 focused tests passed; prior manifest 613/613 unchanged. No register upgrade or independent evaluation. Next: route recovery without requiring an already surveyed return path, and deliberately stress calibration with resets beginning immediately after key pickup. Existing repeated-reset benchmark starts resets after six post-key actions, limiting calibration stress.

Reproduce from root:
```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.probabilistic_calibration_study --seed 98177 --output /tmp/v71a.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.probabilistic_calibration_study --seed 98178 --output /tmp/v71b.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -o addopts='' -q tests/test_probabilistic_calibration.py tests/test_continuous_hazard.py tests/test_hazard_dynamics.py tests/test_adaptive_hazard.py
```
