# v56 — Observability-aware route selection (experimental)

Section 08 remains partial; inherited 13/31 locally verified unchanged.

Diagnostic: seed 85100, 13x13, door column 6, normal controls, clean observations. At action 41 the v50 controller requests rightward displacement from relative (9,5); both images are identical, with horizontal wall and outside-grid band, no anchors. Stationary and palette alignment abstain; visual_displacement returns (0,0), which is ambiguous rather than evidence of no motion. Map alignment also abstains. We designed a route-cost penalty of 8 for edges with fewer than two differing known static cells at corresponding camera pixels; the penalty is heuristic, using observed map only. Dijkstra paths retain inherited blocked-cell and motion verification rules. Candidate is opt-in, no default or protected verifier edits.

## Executed evidence

- Development: seeds 85100–85104, layouts 9:2,11:4,13:6, both control mappings, 0/3/8% noise, identical 65-action cap. 180 trials across two arms: baseline 58/90, candidate 68/90. Ten improved and zero regressed pairs.
- Fresh public seed range 86100–86104, layouts 9:5,11:6,13:6,13:4,15:7, both control mappings, 0/3/8% noise, same cap. 300 trials: baseline 60/150, candidate 70/150; ten improved and zero regressed pairs. Fresh records reproduced byte-for-byte on rerun. The sole improvement is clean 13:6 (0/10 to 10/10, also replicated in development); alternate 13:4 and 15:7 remain unsolved, and 13:6 at 3/8% remains unsolved on fresh seeds. No claim of generalization across geometry or noise; seed repetitions with fixed clean geometry are not independent layouts. More expensive routes can exhaust the action cap. No autonomous learning efficiency measurement or full suite execution.
- 24 focused tests passed. v52 bundle ancestry from f582278 verified and protected verifier directory compared unchanged. v53–55 were source snapshots; their available diagnostic evidence is included. Candidate retained for further evaluation, not promoted as default.

Reproduce from repository root:
```
python -m experiments.observable_route_study --seed-start 85100 --seeds 5 --output dev.json
python -m experiments.observable_route_study --seed-start 86100 --seeds 5 --layouts 9:5,11:6,13:6,13:4,15:7 --output fresh.json
```
Next: distinguish route-budget exhaustion from observation ambiguity; test adaptive replanning and persistent pose hypotheses under actual grid observations, with evaluator-only true pose labels.
