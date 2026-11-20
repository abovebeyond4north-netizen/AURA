# v68: online reset-rate estimation

Experimental fractional Beta updates estimate the probability of a uniform mapping reset, including resets that reproduce the previous mapping. Evidence comes solely from the mapping posterior at certified pose resolutions. This is an approximate online estimator, not exact Bayesian inference: probe steps assume stable dynamics, failures are censored, and the prior is engineered. No independent evaluation or promotion.

Two distinct public seeds (106801, 106802), identical layout configurations, 48 matched cases each. Both controllers achieved 46/48 on each seed and identical failure-adjusted action costs (1140 and 1142). No observed gain or regression; no audited position or mapping errors. Fresh 96 records reproduced byte-for-byte. Nine focused tests passed. All 585 v67 manifest entries matched before packaging. Gymnasium/TensorBoard installation attempted; Gymnasium resolution failed. Section 08 and 03 remain partial; register unchanged (13/31 locally verified per v67).

Reproduce from project root:

```
python -m pytest tests/test_adaptive_hazard.py tests/test_hazard_dynamics.py
python -m experiments.adaptive_hazard_study --seed 106801 --output /tmp/aura68-dev.json
python -m experiments.adaptive_hazard_study --seed 106802 --output /tmp/aura68-fresh.json
```

Next: repeated random rule changes, estimator calibration against evaluator-only event logs, and changed-layout stress; do not infer improvement from parity. Preserve all failures.
