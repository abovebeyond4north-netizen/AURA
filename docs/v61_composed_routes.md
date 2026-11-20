# v61 — Composed early-key and persistent pose controller

Experimental multiple-inheritance composition, with dispatch tests verifying early-key preparation and persistent localization. No policy registration or causal-intervention changes. Development seed 92100 (9:4,11:5,13:8,15:6,17:7); public fresh seeds 93100 (9:3,11:8,13:5,15:9,17:4) and 93200 (9:6,11:3,13:9,15:4,17:10). Both movement mappings, noise 0/3/8%, cap 65. Same evaluator-side pose audit as v60; no private holdout.

| Fresh (60 per arm) | v58 | v60 | v61 |
|---|---:|---:|---:|
| Goals |25|38|36|
| Clean |15|14|15|
| 3% |8|12|14|
| 8% |2|12|7|
| Actions |1093|2494|1433|
| Failure-adjusted cost |2854|2754|2468|
| Trials with audited pose errors |0|0|0|

Against v58: 11 gains, zero losses, identical 579 actions on 25 shared successes. Against v60: three gains, five losses, 759 vs 1134 actions on 33 shared successes. Development goals 16/24/18 respectively out of 30. 270 completed records, 180 fresh records reproduced exactly. 12 focused tests passed. Full suite command timed out after 60% progress marker, no final result; retained partial log. 11 protected files unchanged byte-for-byte. No commit, default promotion or acceptance change. Composition offers a cost/reliability tradeoff rather than dominance over v60. Next investigate 8% failures and full-suite timeout separately.

Reproduce with `python -m experiments.composed_route_study --seed 93100 --layouts 9:3,11:8,13:5,15:9,17:4 --output out.json` (repeat other seeds/layouts above). Tests: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_composed_routes.py tests/test_persistent_pose.py tests/test_efficient_routes.py`.
