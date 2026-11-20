# v64 — Context-dependent dynamics identification (experimental)

The goal-only `ContextDynamicsCamera` extends v63 with a second online identification round after verified door passage. It starts with all 24 cardinal bijections and a known relative position, retains joint pose/mapping hypotheses, probes mapped nondynamic targets, and requires a unique state and at least 12 matching map observations before committing. Unknown/dynamic targets do not justify probes. It retains the previous committed map/mapping on failure and aborts; a failed calibration does not imply the physical agent has not moved. Maximum 12 raw calibration probes and 96 joint states. No simulator labels, true positions, or true mappings enter the policy. The verified-passage trigger is explicitly supplied by the benchmark protocol; unsignaled change detection, arbitrary change times, stochastic/nonbijective effects and neural learning are not implemented. Both arms remove the open door landmark and mark the verified passage cell empty before goal search; the baseline is therefore v63 with this common benchmark hook, not an exact replay of the v63 benchmark. The candidate continues from the resolved pose instead of returning to the calibration start.

## Public matched studies

The evaluator uses identity before collecting the key, a specified permutation after key pickup, and either the same permutation or a different permutation after entering the door. The change is persistent for the rest of the episode. Budget 65 raw actions per episode. Development seed 96000, 9:5 and 11:6, four permutation indices (0,7,16,23), noise 0/8%, 32 trials per controller: v63 16, v64 32 successes. Fresh seed 96100, 9:4 and 13:8, all 24 initial permutations, noise 0/8%: 192 per controller. Stress seed 96200, 11:7 and 15:9, eight indices (0,3,6,9,12,15,18,21), noise 3/15%: 64 per controller. Changed condition uses permutation index (i+11)%24, not all 576 possible pairs. Both arms use the same seed per matched scenario. Public hand-selected layouts and deterministic permutations are not independent secret holdouts.

| Fresh + stress | v63 | v64 |
|---|---:|---:|
| Unchanged controls |120/128|115/128|
| Changed controls |8/128|119/128|
| Total goals |128/256|234/256|
| Actions |5738|8230|
| Failure-adjusted cost (65 for failures) |11394|8702|
| Incorrect audited committed positions |0|0|

111 improved, 5 regressed paired outcomes. Candidate completed 242 post-passage identifications among 244 physically reached passages; all 242 correct. Across both identification stages the candidate completed 498 correct identifications on 256 trials; the audits exclude uncommitted hypotheses, terminal observations and aborted calibrations. Candidate used 1008 second-stage probe actions. On unchanged controls it used 4191 total actions vs 3157, with five lost successes. Thus proactive reidentification has a real cost and is not universally preferable. No significance or general mastery claim. The full architecture needs autonomous change detection and nonbijective/stochastic transition models.

## Reproduction

```
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m experiments.context_dynamics_study --seed 96000 --layouts 9:5,11:6 --indices 0,7,16,23 --output dev.json
python -m experiments.context_dynamics_study --seed 96100 --layouts 9:4,13:8 --indices $(seq -s, 0 23) --noise 0,.08 --output fresh.json
python -m experiments.context_dynamics_study --seed 96200 --layouts 11:7,15:9 --indices 0,3,6,9,12,15,18,21 --noise .03,.15 --output stress.json
python -m pytest -q -o addopts='' tests/test_context_dynamics.py tests/test_learned_dynamics.py tests/test_adaptive_calibration.py tests/test_persistent_pose.py tests/test_composed_routes.py
```

576 original trial records, all 512 fresh/stress records independently rerun byte-identical; 27 focused tests passed. All 534 v63 manifest entries verified before documentation edits; protected verifier unchanged (11/11). No full regression run, Git commit, protected promotion or default policy change. Section 08 partial; overall register remains 13/31 locally verified.
