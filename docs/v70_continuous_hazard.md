# v70 continuous reset propagation
Experimental, not promoted. Propagate reset mass independently at every recovery probe; keep all supported poses and select probes against expanded support. Commit only a unique pose with >=12 map matches. Operational mapping is posterior MAP, not a certified mapping; the legacy mapping audit counts only explicit certifications and initial calibration, NOT operational guesses. The zero mapping-error count therefore cannot be compared as an accuracy improvement. Initial calibration remains unchanged and six cases per controller fail there. The intervention combines reset propagation and relaxed mapping commitment, so causal attribution requires an ablation. No stochastic movement support. Public seeds 98175/98176, 96 trials, fixed baseline 40/48 vs candidate 42/48; two improved, zero regressed; exact two-sided paired sign test p=.5. Zero audited pose errors both. At reset rates 0/.04/.12 successes 14/14/12 baseline vs 14/14/14 candidate (16 each). First seed 48 records byte-identically replayed. Gymnasium import unavailable. Register unchanged (13/31 locally verified). Independent evaluation and full regression pending; protected files unchanged.

Reproduce from root with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1:
```
python -m experiments.continuous_hazard_study --seed 98175 --output /tmp/v70a.json
python -m experiments.continuous_hazard_study --seed 98176 --output /tmp/v70b.json
python -m pytest -o addopts='' -q tests/test_continuous_hazard.py tests/test_hazard_dynamics.py tests/test_adaptive_hazard.py
```
