# v74 — Bounded doorway approach discovery

Base: v73 archive; verified all 643 prior manifest entries. The candidate is additive and opt-in, not default, and leaves the protected verifier, factorial causal capture, and acceptance register unchanged. The goal-only arm remains supplied. When the inherited preparation raises `no isolated approach to the door`, the candidate surveys up to eight reachable viewpoints ranked by unseen map area per route cost, distance to the door, and cost; it only routes through observed walkable tiles, retains switch isolation, and abstains when no route is supported. A known limitation is dependence on the inherited exception message and supplied key-door protocol. No independent evaluation or controller promotion.

## Frozen local matched experiment

Same 65-action cap, immediate resets after key, two starting permutations (indices 2,19), noise 0/.08, reset probabilities 0/.04/.12. Development seed 98203 (previously exposed), geometries (9,3),(11,7). Fresh seed 98205, geometries (9,2),(11,8),(13,5),(13,9), new relative to v73 study, not claimed globally unseen across all project history. Baseline/candidate: development 21/24 vs 23/24; fresh 35/48 vs 36/48; pooled 56/72 vs 59/72. Three matched gains, no regressions; two-sided exact paired sign test p=.25. Failure-adjusted cost 2420 vs 2317 (each failed trial charged 65). Each arm records four mapping errors and zero incorrect audited positions. Mapping checks are sampled at calibration completion, not all movements and not certificates. All three missing-approach failures recovered. Remaining candidate failures: one development reidentified pose cannot return to intended route; twelve fresh continuous hazard pose unresolved. No evidence of stochastic action learning or learning efficiency. Preserve all 144 rows, including failed trials. All 144 replayed byte-identically. Local public evaluation is not an independently administered holdout.

## Regression and dependencies

Focused 17 passed (includes six new tests); general partition 228 passed, 5 skipped, 5 subtests passed, exit 0. Verifier first partition printed 35 passed, 10 deselected, but outer execution timed out, so exit status unconfirmed. Complementary verifier partition 10 passed, 35 deselected, exit 0. No observed failed assertions; do not claim clean whole-suite process completion. Python 3.13.5; Gymnasium and TensorBoard absent; dependency installation not attempted in this iteration. Historical Git metadata/bundle absent from supplied v73 archive, so prior v67 ancestry evidence is carried forward, not rerun. No Git commit or external evaluator access. Register: 13 local verified, 14 partial, 3 open, 1 ongoing; sections 03, 08, 17, 24, 28 and other original outstanding gates remain unchanged.

## Reproduce

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m experiments.frontier_approach_study --seed 98203 --output /tmp/v74-dev.json
python -m experiments.frontier_approach_study --seed 98205 --expanded --output /tmp/v74-fresh.json
cmp /tmp/v74-dev.json verification/section08/v74/development.json
cmp /tmp/v74-fresh.json verification/section08/v74/fresh.json
python -m pytest -o addopts='' -q tests/test_frontier_approach.py tests/test_door_route_recovery.py tests/test_probabilistic_calibration.py tests/test_continuous_hazard.py
python -m pytest -q -o addopts='' --ignore=tests/test_verifier.py
python -m pytest -q -o addopts='' tests/test_verifier.py -k 'not (seccomp_denies_legacy_creat_syscall or seccomp_denies_write_capable or seccomp_denies_memfd or seccomp_denies_openat2 or seccomp_denies_xattr or runner_)'
python -m pytest -q -o addopts='' tests/test_verifier.py -k 'seccomp_denies_legacy_creat_syscall or seccomp_denies_write_capable or seccomp_denies_memfd or seccomp_denies_openat2 or seccomp_denies_xattr or runner_'
```

Next: diagnose large-layout visual aliasing and recover from ambiguous beliefs without unsafe localization commitments; learn stochastic action effects rather than assuming a deterministic permutation. Retain existing independently controlled promotion requirements.
