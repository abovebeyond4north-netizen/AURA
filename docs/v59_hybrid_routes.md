# v59 — Initial-image strategy selector (rejected)

## Intervention
`HybridRouteCamera` chooses v58 early-key pickup if at least 11 wall-colored pixels are visible in the first RGB frame; otherwise it uses v57 survey-first. The threshold was chosen after inspecting v58 public development outcomes and initial-image wall counts (not learned by the agent). Neither the environment size, door coordinate, reversal flag, noise setting, seed, nor trial outcome is supplied to the selector. This is a one-time choice, not online adaptation or a general estimator of observability. Both branches retain the same verified motion procedure and 65-action cap. Only goal arm 2 is supported. No modification to the causal measurement controller or default policy.

## Fresh public evaluation
Six seeds: 90100–90102 on 9:3,11:5,13:5,15:6,17:7; 90200–90202 on 9:6,11:3,13:7,15:4,17:9. Layout entries are size:door column. Each layout crosses normal/reversed movement and 0%, 3%, 8% independent pixel corruption. Matched reset and cap; actual noise streams can diverge with interaction count. 90 cases per controller per batch, 540 completed records overall. Fresh seeds are public; repeated geometries are not independent samples of the full environment family.

| Batch | v57 goals/90 | v58 goals/90 | v59 goals/90 |
|---|---:|---:|---:|
| A | 46 | 43 | 47 |
| B | 46 | 47 | 40 |
| Total/180 | 92 | 90 | 87 |

| Noise | v57/60 | v58/60 | v59/60 |
|---|---:|---:|---:|
| 0% |42|45|42|
| 3% |28|26|26|
| 8% |22|19|19|

Total raw actions 5865/3506/5633; failure-penalized (65 per failure) 8484/7676/8562. Hybrid improved one and regressed six cases vs v57; improved four and regressed seven vs v58. On 86 jointly successful v57/v59 trials costs were 2628/2502; on 83 jointly successful v58/v59 trials costs were 1589/2389. Selector chose early pickup in 34/180 trials, survey first in 146/180. These outcomes reject the hypothesis that this threshold reliably combines the strengths of the controllers. No post-hoc retuning on the evaluation cases. No significance or general sample-efficiency claim.

## Verification
540/540 stored trial records reproduced byte-for-byte in six independent process shards. 35 focused tests passed, including selector dispatch, no extra sampling, invalid pixel handling and motion contract. All 11 protected-verifier files unchanged compared to v58. No Git metadata in supplied v58 archive; no new commit, promotion, full regression run or acceptance change. Default v50 unchanged; retain v57 and v58 as distinct research tradeoffs. Section 08 partial; 13/31 inherited local statuses remain.

Reproduce: `python -m experiments.hybrid_route_study --seed 90100 --layouts 9:3,11:5,13:5,15:6,17:7 --output run.json`; repeat all seeds using the above layout groups. Focused test command in verification/section08/v59/tests_command.txt.

Next: prioritize persistent motion hypotheses or directly measure predicted visibility of calibration and survey edges before inferring controller suitability; use training/development cases for selection and new independently reserved geometries for assessment. Do not reuse these outcomes as confirmatory evidence.
