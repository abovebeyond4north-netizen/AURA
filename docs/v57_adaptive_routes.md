# v57 — Risk-aware adaptive survey (experimental)

Section 08 remains partial; 13/31 inherited local acceptance statuses unchanged. Source is the v56 archive, not a new Git commit.

## Diagnosis and intervention

Replayed v56 cases at seed 86100. On 13:6 at 3% noise, motion at action 6 loses the last visible switch anchor; stationary alignment and palette alignment abstain, while generic visual displacement returns `(0,0)`. Only one mapped stationary contrast differentiates the hypotheses. On clean 15:7, action 7 produces identical anchorless images. On clean 13:4, v56 exhausts the 65-action cap. The v56 path finder penalizes visual aliasing, but inherited survey destination ranking divides information gain by unweighted route length and executes a whole path without re-evaluating new observations.

`AdaptiveRouteCamera` retains the v56 graph, cost penalty (8 for fewer than two mapped static contrasts), strict motion verification, calibration, passage and post-passage goal search. It ranks unvisited survey targets by estimated unknown cells per risk-weighted path cost and executes only the first edge before re-planning. Returning home still uses a fixed known path. No hidden coordinates, ground truth, learned policy, new pose recovery or modified protected verifier. The common harness uses the 65-action cap, normal/reversed controls, and same seeds/layouts/corruption in each arm. These are public seeded evaluations, not independently held-out verification. Geometry is fixed per size/column; repeated clean trials are not independent maps.

## Results

Development: seeds 86100–86102; layouts 9:5,11:6,13:6,13:4,15:7,15:3; baseline 42/108 vs candidate 58/108 (16 improved, zero regressed). Total actions 1,938 vs 3,670.

Fresh public seeds 87100–87104; layouts 9:2,11:4,13:6,13:4,15:7,15:3,17:8; baseline 62/210 vs candidate 98/210 (42 improved, six regressed). 420 trial records reproduced byte-for-byte. Across both batches, 636 executed trial records, excluding replay. Candidate executes 7,716 vs baseline 3,238 actions in fresh batch; both capped at 65 per trial. Higher total success does not imply learning/sample efficiency. No new default/promotion; retain v56 for comparison and v50 as established controller. Clean larger-layout results improve, but 15:3 remains unsolved and noisy larger layouts mostly fail. All six regressions occur on 11:4 noisy trials. All 112 fresh candidate failures are movement-contract failures, not action-cap exhaustion. Full suite not run. 27 focused tests passed. Protected verifier file contents compared byte-for-byte by SHA-256 against v56 (11 files); unchanged.

Reproduce:
```
python -m experiments.adaptive_route_study --seed-start 86100 --seeds 3 --layouts 9:5,11:6,13:6,13:4,15:7,15:3 --output dev.json
python -m experiments.adaptive_route_study --seed-start 87100 --seeds 5 --layouts 9:2,11:4,13:6,13:4,15:7,15:3,17:8 --output fresh.json
python -m pytest -q -o addopts='' tests/test_adaptive_routes.py tests/test_observable_routes.py tests/test_goal_stationary_alignment.py tests/test_palette_goal.py tests/test_recovery_goal.py tests/test_passage_goal.py tests/test_map_goal.py tests/test_active_goal.py tests/test_pose_belief.py
```

Next: isolate regression-causing routes with actual observations, test multi-step pose hypotheses and selection policies with capped probing, compare interaction cost and success on matched public and independently held-out layouts; no claims of autonomous learning gain yet.
