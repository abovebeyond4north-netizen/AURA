# Section 08 — Visual-frontier exploration continuation (v4)

**Status: locally verified physical exploration improvement; Section 08 remains PARTIAL.**  The benchmark does not demonstrate broad control mastery or a simultaneous held-out prediction advantage. All numbers below are from actually executed policies, not hand-coded goal trajectories.

## Problem and falsifiable hypothesis

Previous self-supervised validated-progress curiosity did not outperform matched random exploration on full image-gridworld coverage. Hypothesis: spatial exploration driven by **visually inferred movement** and an observed action-transition graph can explore more controllable space than unconstrained random actions within the same interaction budget, without privileged simulator state.

## Mechanism and priors

- `aura/exploration/visual_frontier.py`: `visual_displacement` aligns consecutive 7×7 RGB image views under possible one-pixel camera translations. A translation is accepted only when the overlap exceeds a minimum match and exceeds the unshifted match by a margin. All experience comes from observed images and chosen actions.
- The resulting online pose is *relative* to the episode start. A graph records **observed** directed action effects. Frontier selection prefers actions leading to unvisited inferred cells; BFS returns to reachable frontiers. The graph resets at each episode. There is no externally supplied true coordinate, map, goal location or solution sequence.
- Engineering priors are disclosed: images are egocentric, camera translations are local/cardinal, and a declared set of discrete action IDs can move the agent. A separate action ID activates nearby mechanisms. These priors are not autonomously discovered.
- Visually unusual neighboring colors trigger interaction probes; they are detected from generic pixel frequency, **not** labeled as keys/switches/doors. A cooldown prevents immediate repeat interaction that would reverse a toggle. Interventions selected from the main agent's policy are tracked as externally selected actions, preserving odometry.
- Extra nonnavigation actions are probed several times; later probes are only scheduled for actions with repeatable stable-pixel effects. In this environment the independent TV action remains unproductive. This is a heuristic and can miss a useful action whose effect appears only in another context.
- The action-conditioned learning-progress model still updates its frozen local validation bank, but frontier selection does **not** solve value-of-information or train a goal-directed value function. The primary model continues ordinary PyTorch optimization from random initial weights.

## Design and changes after development feedback

1. First exploratory controller was evaluated on seeds 0–4 and an early confirmation batch 20–29. In those batches, a repeated interaction could turn a switch off after turning it on. Those results were used to correct the controller and are **not valid frozen confirmation** for the revised version.
2. The revised controller was then evaluated without further policy changes on **fresh seeds 30–39**, 500 interactions per policy, per task. Each seed/task pair received identical simulator task distributions and budgets. The agent saw the same image/action/reward interface for each policy. Evaluator-only `diagnostic_state()` provided absolute physical coverage and success statistics; it was never read by the policy.
3. The policies were random actions, raw prediction error, the earlier validated-progress policy, and the new visual-frontier policy. Results were recorded per seed, and `experiments/analyze_frontier.py` checked pairing/duplicate records before computing 95% Student-t intervals for paired differences. Three task families were inspected; intervals are **not corrected for multiple comparisons**, so they are exploratory rather than familywise statistical guarantees. Seeds are local and not cryptographically protected.

## Fresh-seed final results (10 seeds 30–39; 500 interactions per seed)

Number of distinct *physical* grid positions visited across the 500 interactions (using privileged evaluation only; not part of observation):

| Task | Random | Error curiosity | Validated progress | Visual frontier | Paired gain vs random, 95% CI |
|---|---:|---:|---:|---:|---|
| Key | 21.9 | 13.7 | 19.9 | **34.0** | +12.1 [9.63, 14.57] |
| Switch | 23.6 | 14.1 | 20.9 | **35.2** | +11.6 [8.21, 14.99] |
| Noisy-TV | 21.6 | 13.0 | 21.5 | **36.4** | +14.8 [10.78, 18.82] |

The older episode-position metric also improved: 200.7 vs 120.9 for key, 200.5 vs 122.5 for switch, and 218.7 vs 101.9 for noisy TV. It counts `(episode ID, position)` pairs, **not** unique world cells; report the physical-position metric for map coverage.

| Task | Random wins (ten 500-step runs) | Visual-frontier wins | Random held-out feature MSE | Visual-frontier held-out feature MSE |
|---|---:|---:|---:|---:|
| Key | 0 | 0 | 0.0031355 | 0.0031207 |
| Switch | 0 | 1 | 0.0030973 | 0.0030400 |
| Noisy-TV | 0 | 1 | 0.0025649 | **0.0029622 (worse)** |

On the final half of the noisy-TV episodes, random selected the unproductive sixth action 17.2% of the time; visual frontier selected it 0%. This reflects three initial probe actions per experiment and subsequent suppression when effects do not recur; it **does not** establish robust general discrimination of irreducible noise. The policy may ignore a contextual action whose early probes were unproductive.

**Acceptance conclusion:** A reproducible increase in spatial exploration coverage was confirmed on new seeds. Held-out prediction is not consistently better, with a clear noisy-TV regression, and total successful task completions (2 vs 0) are too rare to substantiate generally improved control/sample efficiency. Section 08 stays partial, with a new positive engineering sub-gate.

## How to reproduce

```bash
pytest
python -m experiments.gridworld_exploration --seeds 10 --seed-start 30 --steps 500 --output runs/gridworld_exploration_v4.json
python -m experiments.analyze_frontier sample_outputs/visual_frontier_v4_seeds_30_34.json sample_outputs/visual_frontier_v4_seeds_35_39.json --output runs/visual_frontier_recomputed.json
python scripts/train.py --config configs/visual_frontier_cpu.yaml --condition curiosity --task key
```

The saved report is `sample_outputs/visual_frontier_v4_statistical_report.json`. The original 76 tests remain and additional new tests cover visual odometry, altered action mappings, independent TV probing, external-action synchronization, budget comparison validation, and integrated evidence logging.

## Next falsification gates

1. Demonstrate *goal completion or measured held-out learning efficiency*, not just broader coverage, on fresh unseen task families under matched budgets.
2. Learn action roles and affordances contextually; avoid fixed move-vs-interact action-prior advantages and detect when a previously unproductive action becomes useful.
3. Correct visual odometry aliasing under repeated textures, rotation, stronger noise, and camera scale changes; quantify localization failure and recovery.
4. Evaluate a learned high-level skill planner over the discovered transition graph, then test transfers and forgetting across key, switch, altered, and hidden-mechanism families.
5. Account for compute/energy overhead and preregister independent protected evaluations before promotion.

These limitations make this a scoped exploration algorithm, not evidence of unrestricted autonomous intelligence.
