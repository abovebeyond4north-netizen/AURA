# v38 cross-section engineering and acceptance map

Parent `af3c192`; original ancestor `f582278`. The register statuses are unchanged: four locally verified, 23 partial, three open, one ongoing. This table identifies concrete changes and evidence or an explicit remaining gate for every section. No independent promotion is authorized by the development experiments.

| Section | v38 implementation/evidence | Remaining gate |
|---|---|---|
|01 Mission | Opt-in reward-route learner completes real key episodes and repeats rewarded actions | General, neural, cross-layout and recursive learning gains |
|02 Requirements | Existing protected policy unchanged; four opt-in namespace tests rerun; protected study fails closed | Private dependencies, isolation, legitimate evaluator workload |
|03 Stack | Local CPU test and import/compile evidence | Full supported dependency/platform/GPU matrix |
|04 Architecture | Integrated task-local feedback, persisted actor, reward-route policy | Independent end-to-end acceptance |
|05 Perception | Existing encoder tests rerun; new observational matching policy uses image inputs | More varied real observations and acquired representations |
|06 World model | Continued temporal updates across interrupted training; adaptive and route runs | Transferred abstract dynamics and held-out task model advantage |
|07 Uncertainty | Existing calibration tests rerun; task-local error histories avoid mixing task validation losses | Calibration under hidden mechanisms and noise |
|08 Curiosity | Frontier reopens previously obstructed edges; matched route and random exploration studies | Replication on changed layouts and noise robustness |
|09 Causal discovery | Integrated visual-change hypothesis labeled observational, not randomized | Actual randomized interventions and identification in primary environment |
|10 Evidence | Gain pairs and route candidates reference observed ledger records; actual ledgers verified | External immutable anchor and independently secured evidence |
|11 Concepts | Seven candidates accepted by local criterion in adaptive run | Novel-task transfer, split/merge/retirement and independent testing |
|12 Skills | Distinct successful episode support and monotonic IDs; bounded route candidate retry and abort, 21 exact reproductions in comparative study | Reuse on unseen layouts/mechanisms, independent option evaluation |
|13 Curriculum | Unattempted-task coverage before repetition; actual five-family adaptive execution | Matched multi-seed curriculum advantage |
|14 Continual learning | Live segmented training matches uninterrupted execution in regression | Durable complete restart, gridworld retention/transfer |
|15 Meta-learning | Honest action-policy mode names, sparse signed paired gains, per-mode schedule; every executed recurring mode receives measurements | Causal strategy attribution and independently validated adaptive advantage |
|16 Self-improvement | Existing study runs and fails closed | Two independently accepted generations |
|17 Verifier | No protected files changed; opt-in namespace probes pass; real smoke blocked | Complete independent sandbox and legitimate evaluation |
|18 Computation | Per-family development interaction accounting; distinct missing and measured gains; cumulative live-run timing | Matched total costs and empirically optimized allocation |
|19 Environment | Five task control comparisons; altered/hidden route success; noise fails | Novel procedural topology and richer mechanisms |
|20 Structure | Added modular policy, development evaluator, experiment and tests | Maintain existing verified structure |
|21 Typed interfaces | Added bounded route contract and actor checkpoint schema; existing contracts rerun | More adversarial typed-contract checks |
|22 Core loop | Preserves partial episodes in memory; reward-route opt-in with real evidence IDs | Integrating options with adaptive policy without unvalidated promotion |
|23 Objectives | Fresh paired prediction deltas, no stale credit; actor roundtrip and control evaluation | Objective that learns effective neural goal control |
|24 Benchmarks | 3-seed matched exploration and route studies, paired development actor comparator | Preregistered holdout, more seeds, changed layouts, confidence intervals |
|25 Recursive validation | Actual protected smoke blocked at first evaluation | Two independently validated improvements |
|26 Phases | Integrated two-run study with evidence artifacts and explicit open gates | Meet all remaining phase exit criteria |
|27 CLI/artifacts | `scripts.cross_section_study`, `scripts.evaluate_control`, `experiments.reward_route_study` | End-to-end reproducible supported-platform qualification |
|28 Quality | 149 default tests pass, 4 opt-in namespace tests separately pass; compilation passes | Wider adversarial and OS matrix, independent reviews |
|29 Workflow | Versioned reproducible logs, tests and code, local bundle | Remote review/CI not performed |
|30 Final acceptance | Explicitly open; positive bounded results are not final certification | All dependent exit gates |
|31 Principles | Failed baselines retained; no protected policy changes or candidate promotion | Continue evidence-driven, fail-closed research |

## Reproduction

```sh
python -m pytest -q -o addopts=''
AURA_TEST_NAMESPACES=1 python -m pytest -q -o addopts='' tests/test_namespace_probe.py
python -m experiments.reward_route_study --seed-start 90 --seeds 3 --steps 640 --horizon 80 --output NEW_REWARD_REPORT.json
python -m scripts.cross_section_study --output NEW_STUDY_DIRECTORY
python -m scripts.evaluate_control --checkpoint NEW_STUDY_DIRECTORY/adaptive/checkpoint.pt --episodes 5 --max-steps 40 --seed-index 38 --output NEW_CONTROL_REPORT.json
python -m scripts.minimum_architecture --output NEW_SMOKE_DIRECTORY
```

The last command is expected to exit 2 in this environment because legitimate recursive evaluation remains blocked by the temporary-directory security restriction. `reward_routes` is experimental and explicitly requires `condition='curiosity'`; it is not the default adaptive controller or a protected candidate. `train(steps=N)` now accepts an integer cumulative target within configuration bounds; changing the task or condition mid-episode is rejected. Live continuation is in-process only, not a process-restart checkpoint. `actor.pt` preserves neural actor and optimizer state, not all learner state or route memory. Actor policy evaluation measures a greedy neural actor, not the combined exploration policy. No independent test gain is claimed. The development seed partition is public and deterministic (not a secrecy guarantee); the per-task online banks share the configured environment seed but have separately generated data and provenance. Meta gains are paired model-update deltas, not causal attribution to action-selection policies. Candidate skills mined from successful sequences are not validated options, and reward routes use image similarity with structural priors, not invented semantic concepts. The fixed-layout route experiment uses 3 seeds per task; the integrated key run uses a diagnostic seed that was chosen after earlier results and is not a confirmatory evaluation. No goal was attained in the noise family or by the separate neural actor. No full hostile-code security certification is claimed.
