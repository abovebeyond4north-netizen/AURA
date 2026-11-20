# v65 — Observation-triggered movement-dynamics reidentification

Experimental, no promotion. Starts with v64 source; all 545 v64 manifest hashes matched before edits. Three research mechanisms: `ChangeDynamicsCamera` monitors post-key motion against all 24 bijections; accepts a unique supported expected pose, otherwise freezes the map and uses up to 12 raw probes to identify a joint pose/mapping with at least 12 matching map observations. `EvidenceChangeCamera` adds stationary landmark and exact/palette alignment evidence (same development outcomes). `ReplanningChangeCamera` replans to the original destination after recovery rather than necessarily returning to the aliased edge. The initial key-pickup calibration remains externally structured. The change detector receives no timing, passage notification, hidden mapping or diagnostic position. The common goal benchmark still marks the observed open doorway after measured passage. Known free tiles are assumed deterministic, unknown/dynamic tiles allow blocked or moved outcomes; the mapping is assumed constant during a recovery window. This is finite constrained hypothesis identification, not calibrated Bayesian/neural prediction or general environmental-rule discovery. Failure aborts rather than silently assigning unresolved poses; physical movement may already have occurred. Even correct completed identifications may be unnecessary recalibrations, not necessarily detections of actual changes.

## Matched public results

65 raw action cap. The evaluator applies identity until key pickup, then a fixed specified permutation. `none` keeps the mapping; `door` changes after physically entering the doorway; `late` changes after 10 post-key actions, without notifying the agent. Changed mapping index `(i+11)%24`. Tests use selected public geometries and seeds, not independent holdouts. v63 and v64 are exercised with the common verified-passage goal protocol; v64 alone gets the engineered recalibration hook. The same exogenous schedule and seed are applied to each matched arm; because policies consume different actions, the timed change may occur at different locations. Audit checks `_observe` pose commitments and completed calibration mappings, not terminal observations or uncommitted hypotheses.

| Fresh+stress | v63 | v64 | v65 strict |
|---|---:|---:|---:|
| Unchanged |43/48|37/48|24/48|
| Door change |8/48|37/48|24/48|
| Timed change |6/48|12/48|22/48|
| Total|57/144|86/144|70/144|
| Raw actions|3113|4302|4238|
| Failure-adjusted actions|7245|6635|6537|
| Incorrect committed poses|2|2|0|

Relative to v64, 12 improved and 28 regressed paired outcomes; 169/169 completed joint reidentifications matched the actual mapping, including unchanged-map recalibrations. 66/74 strict failures were repeated uncertainty returning to the route, six lacked isolated approaches, two lacked a return route. No promotion. Replanning variant on a separate public larger-layout batch (seed 97300) 8/48 vs strict 6/48, v64 17/48, v63 14/48; under timed change 4/16 vs 2/16 strict vs 0/16 v64. Replanning did not remedy unchanged-control losses (2/16). The evidence-augmented variant was only run on development cases and had identical success outcomes to strict. Future priority: amortize localization via safe route-level active belief planning with a stable-mapping hypothesis and a change hazard, rather than forcing full reidentification on every aliased edge. No claim of confidence calibration or arbitrary change detection.

## Reproduction

Run from source root with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Output paths must not exist.

```
python -m experiments.change_dynamics_study --seed 97000 --layouts 9:5,11:6 --indices 0,7,16,23 --noise 0,.08 --output dev.json
python -m experiments.change_dynamics_study --seed 97100 --layouts 9:4,13:8 --indices 1,4,8,12,17,22 --noise 0,.08 --output fresh.json
python -m experiments.change_dynamics_study --seed 97200 --layouts 11:7,15:9 --indices 2,5,9,13,18,21 --noise .03,.15 --output stress.json
python -m experiments.change_dynamics_study --seed 97000 --layouts 9:5,11:6 --indices 0,7,16,23 --noise 0,.08 --enhanced --output enhanced_dev.json
python -m experiments.change_dynamics_study --seed 97000 --layouts 9:5,11:6 --indices 0,7,16,23 --noise 0,.08 --replan --output replan_dev.json
python -m experiments.change_dynamics_study --seed 97300 --layouts 13:7,15:8 --indices 0,6,12,18 --noise 0,.08 --replan --output replan_fresh.json
python -m pytest -q -o addopts='' tests/test_change_dynamics.py tests/test_context_dynamics.py tests/test_learned_dynamics.py tests/test_adaptive_calibration.py tests/test_persistent_pose.py tests/test_composed_routes.py
```

1152 original trial records across six runs (including repeated development controls); 624 fresh/stress/replan-fresh records replayed byte-for-byte. 33 focused tests passed. The initial timed-change unit test selected a failed mapping/layout and was corrected to explicitly test a successful development fixture; failed cases remain in the trial records. No full regression suite executed. No new Git commit, bundle or default policy change. Section 08 partial; inherited register 13/31 verified locally, not revalidated across all sections here.
