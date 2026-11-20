# v66 — Hazard-weighted dynamics identification (experimental)

The v65 source was restored and all 559 original manifest entries checked. The new `HazardDynamicsCamera` maintains a normalized distribution over the 24 engineered cardinal bijections, using a fixed 3% uniform-resampling hazard at each monitored movement. Predicted moved/blocked outcomes receive equal mass when a target is unknown; incompatible static scene hypotheses are eliminated and the survivors renormalized. The prior affects probe selection through weighted pairwise predicted scene contrast, not acceptance of an ambiguous pose. A unique pose with at least 12 matching mapped observations and continued compatibility of the current mapping can be committed without identifying the entire mapping. A changed mapping is committed only for a singleton joint state. The shared map is not updated during unresolved probes. The existing route-replanning mechanism is inherited. As in v65, post-key calibration is externally structured and movement dynamics are assumed fixed within each bounded recovery window. This is not a learned hazard rate, a calibrated observation likelihood, arbitrary stochastic dynamics discovery, or independent research validation. The simulator's off-palette corruption is treated as missing visual evidence; in-palette corruptions and dynamic scenery can violate the model.

## Matched public evaluation

Same 65-action cap and evaluator-only pose and mapping audit as v65. Changes occur at doorway crossing or after ten post-key actions; no change notification is supplied to v66. Fixed code and hyperparameters were used for development, fresh and stress batches. The primary comparison uses v64, v65 strict, and v66. After inspecting the primary results, two additional same-case ablations were run: v65 route replanning and v66 with a uniform (hazard=1) prior. These are diagnostic comparisons on the same cases, not additional independent test sets.

| Fresh + stress (96 cases per policy) | Stable / 32 | Door / 32 | Timed / 32 | Goals | Raw actions | Failure-adjusted actions | Audited pose/mapping errors |
|---|---:|---:|---:|---:|---:|---:|---|
| v64 context |19|19|5|43|2076|4837|0/0|
| v65 strict |11|11|12|34|1994|4853|0/0|
| v65 replanning |17|17|12|46|2781|4575|0/0|
| v66 uniform |17|17|13|47|2514|4521|0/0|
| v66 weighted |23|22|21|66|2370|4166|0/0|

Failure-adjusted actions charge 65 for every unsuccessful case. Relative to v65 strict, v66 improves 32 cases and loses none; versus the uniform ablation it improves 20 and loses one. Weighted and strict reach 39/48 on development. The weighted candidate reaches 44/48 fresh and 22/48 stress. It uses 302 change probes and reports 43 completed changed-mapping commitments on the combined fresh/stress cases, all correct per evaluator audit. The audit does not cover terminal frames or hypothetical states. The 96 cases use two public seeds with repeated geometries, permutations and corruption settings, so no independence-based significance claim is made. The weighting is a hand-set prior, not learned from data; no default controller promotion.

## Reproduction

Run from source root with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Use new output paths:

```
python -m experiments.hazard_dynamics_study --seed 97000 --layouts 9:5,11:6 --indices 0,7,16,23 --noise 0,.08 --output dev.json
python -m experiments.hazard_dynamics_study --seed 97400 --layouts 9:4,13:8 --indices 1,4,8,12 --noise 0,.08 --output fresh.json
python -m experiments.hazard_dynamics_study --seed 97500 --layouts 11:7,15:9 --indices 2,5,9,13 --noise .03,.15 --output stress.json
python -m experiments.hazard_dynamics_study --seed 97400 --layouts 9:4,13:8 --indices 1,4,8,12 --noise 0,.08 --ablation --output fresh_ablation.json
python -m experiments.hazard_dynamics_study --seed 97500 --layouts 11:7,15:9 --indices 2,5,9,13 --noise .03,.15 --ablation --output stress_ablation.json
python -m pytest -q -o addopts='' tests/test_hazard_dynamics.py tests/test_change_dynamics.py tests/test_context_dynamics.py tests/test_learned_dynamics.py tests/test_adaptive_calibration.py tests/test_persistent_pose.py tests/test_composed_routes.py
```

624 original trial records, including 144 development records. All 480 fresh/stress and ablation records exactly replayed. 39 focused tests passed. All 11 protected verifier files match v65. No Git commit or full regression suite; independent holdout outstanding. Section 08 remains partial; inherited 13/31 register not independently revalidated across all sections.
