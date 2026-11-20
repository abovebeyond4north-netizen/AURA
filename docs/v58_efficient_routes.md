# v58 — Early-key goal-only route optimization (experimental)

## Intervention

The v57 controller inherits a factorial door-test procedure: survey; return to start; collect key; calibrate; return to start; sham wall action; approach the door. This is appropriate for comparing interventions, but redundant for a key-goal-only policy. `EfficientRouteCamera` takes the observed adjacent key first, retains verified movement and calibration, surveys from the key location, and approaches the observed door without the sham or return-to-home detour. It uses the same route-risk ranking and post-passage search. It explicitly rejects arms other than 2 and must not be used for factorial causal estimation. The key mechanism, color palette, camera geometry and action priors are engineered. No hidden simulator coordinates or diagnostic state are consulted. No learning or persistent pose estimation is added.

## Experimental design

Three controllers on equal 65-action caps, same seeded reset, noise and initial geometry per matched triple (noise stream depends on interaction history). Development: seeds 88100–88101, layouts 9:5,11:6,13:6,13:4,15:7,15:3,17:8; normal/reversed controls and 0%, 3%, 8% independent pixel corruption. Fresh public assessment: seeds 89100–89104, layouts 9:2,11:4,13:6,13:4,15:7,15:3,17:8,17:5. The same geometry is repeated across seeds; clean repeats do not establish independent map generalization. The source snapshot is v57 (no Git metadata or new commit). Two long combined runs timed out before writing complete results; smaller per-seed shards completed, and all five fresh shards later reproduced byte-for-byte. Exclude incomplete attempts and replay from experimental totals.

| Batch | Cases per arm | v56 wins | v57 wins | v58 wins | v56 raw actions | v57 raw actions | v58 raw actions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Development | 84 | 26 | 38 | 36 | 1354 | 2570 | 1811 |
| Fresh | 240 | 58 | 100 | 90 | 3444 | 7658 | 5202 |

Fresh wins by noise (80 per controller per condition): v56 30/16/12; v57 60/26/14; v58 55/25/10. Matched to v57, v58 improves seven trials and regresses seventeen; 83 jointly successful trials use 2962 versus 1859 actions. This conditional metric is not general sample efficiency. Failure-penalized cost (each failure charged 65 regardless of early termination) is 13356/12834/11764 for v56/v57/v58. Against v56, v58 improves 36 cases and regresses four. All 150 fresh v58 failures are motion-contract failures. In particular, all 15:3 and 17:5 trials remain unsolved. Fresh seeds are public, not independent verifier assets. No inferential significance claim.

## Verification and eligibility

972 completed trial records across both batches (252+720), excluding initial exploratory probes, timed-out attempts and replay. All 720 fresh records reproduced exactly in five shards; 30 focused tests passed. 11 protected-verifier files compared byte-for-byte against v57, unchanged. No new default policy, no Git commit, no full regression run, no acceptance promotion. Keep v50 established and v57 research comparison; v58 presents a cost/success tradeoff rather than a dominating candidate. Section 08 partial; 13/31 inherited bounded local statuses unchanged.

## Reproduction

```
python -m experiments.efficient_route_study --seed-start 88100 --seeds 2 --layouts 9:5,11:6,13:6,13:4,15:7,15:3,17:8 --output dev.json
for seed in 89100 89101 89102 89103 89104; do
  python -m experiments.efficient_route_study --seed-start "$seed" --seeds 1 --layouts 9:2,11:4,13:6,13:4,15:7,15:3,17:8,17:5 --output "fresh_$seed.json"
done
python -m pytest -q -o addopts='' tests/test_efficient_routes.py tests/test_adaptive_routes.py tests/test_observable_routes.py tests/test_goal_stationary_alignment.py tests/test_palette_goal.py tests/test_recovery_goal.py tests/test_passage_goal.py tests/test_map_goal.py tests/test_active_goal.py tests/test_pose_belief.py
```

Next: persistent motion hypotheses with explicit disambiguation, test whether local observations predict reliable route selection, and compare joint success and failure-penalized cost without using evaluation outcome to choose a policy per trial.
