# v63 — Online action-effect identification (experimental)

## Mechanism and assumptions

The goal-only camera starts with 24 bijections of raw action IDs to four cardinal displacements after the key pickup event, rather than the earlier explicit binary reversal. It maintains joint mapping/pose hypotheses, considers both blocked and moved outcomes, filters using stationary visual overlap and mapped tiles, and selects raw probes by weighted distinguishable map-view pairs (entropy tie-break). No hidden coordinates or control labels are supplied to the policy. Unknown/dynamic targets are excluded from extra probes; no unresolved hypothesis writes into the shared map. Unique mapping and pose plus 12 map matches are required before committing. At most 12 probes and 96 joint hypotheses are allowed. Before pickup, identity control is assumed. After pickup, the learned inverse mapping routes subsequent movement commands. All four cardinal moves, bijectivity, the pickup-triggered change, known palette, static tile compatibility, and deterministic movement are engineered priors; this is bounded online model identification, not a neural model trained from scratch, calibrated probabilistic learning, arbitrary action discovery, or general causal learning. The planner and goal protocol remain engineered. No candidate promotion.

## Matched evaluation

`experiments.learned_dynamics_study` wraps the existing key gridworld. The evaluator remaps raw motion after key pickup; the learner receives only RGB observations and the step callback. The evaluator audits committed position and completed mappings. All trials have a 65-action cap, same initial seed per matched comparison. The baseline is v62 CalibratedComposedCamera (two possible mappings); the candidate is LearnedDynamicsCamera (24). This benchmark intentionally expands the hypothesis class, so the large result is evidence of control-mapping coverage, not a fair model-capacity-controlled comparison of learning efficiency. The synthetic layouts are hand chosen and public, not independent secret holdouts.

Development: seed 95100, layouts 9:5,11:6, mapping indexes 0,7,16,23, noise 0/8%, 16 trials per arm: baseline 8, candidate 16 goals, no audited position errors. Fresh: seed 95200, layouts 9:4,11:7,13:8, all 24 permutations, noise 0/8%, 144 per arm: baseline 12, candidate 144. Stress: seed 95300, layouts 9:6,13:4,15:9, all 24, noise 3/15%, 144 per arm: baseline 11, candidate 103.

| Metric (fresh + stress only) | v62 | v63 |
|---|---:|---:|
| Goals |23/288|247/288|
| Completed mappings |48|288|
| Incorrect completed mappings |24|0|
| Incorrect committed positions (audit records) |0|0|
| Raw actions |1475|7265|
| Failure-adjusted cost (65 for failures) |17794|8734|
| Goals on original two mappings |23/24|22/24|
| Goals on other 22 mappings |0/264|225/264|

226 paired improvements, two paired regressions. At 15% noise candidate reached 37/72 goals, despite complete correct mapping identification. All 288 completed candidate calibration calls identified the correct mapping. The audit excludes aborted/intermediate hypotheses and is not proof in general. No independent holdout or significance claim. 608 original records including development; 576 fresh/stress records independently rerun byte-identical. 22 focused tests pass. Full repository suite not executed. No new Git commit or default controller change. Acceptance remains 13/31, Section 08 partial. Next: address downstream movement and planning failures under higher corruption while measuring efficiency on the original mapping family and testing model assumptions (non-bijective or state-dependent control).

Reproduction (run from source root):

```
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m experiments.learned_dynamics_study --seed 95100 --layouts 9:5,11:6 --output dev.json --mapping-indices 0,7,16,23
python -m experiments.learned_dynamics_study --seed 95200 --layouts 9:4,11:7,13:8 --output fresh.json --mapping-indices $(seq -s, 0 23) --noise 0,.08
python -m experiments.learned_dynamics_study --seed 95300 --layouts 9:6,13:4,15:9 --output stress.json --mapping-indices $(seq -s, 0 23) --noise .03,.15
python -m pytest -q -o addopts='' tests/test_learned_dynamics.py tests/test_adaptive_calibration.py tests/test_persistent_pose.py tests/test_composed_routes.py
```
