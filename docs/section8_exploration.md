# Section 08 — visual curiosity: implemented diagnostic and limits

This document accompanies commit after `138be47`. The algorithm is **experimental and Section 8 is PARTIAL**. No successful broad gridworld-learning or sample-efficiency superiority over random has been established. Benchmark summaries must not be described as statistically significant results merely from five seeds.

## Intervention and provenance

The training agent only receives `uint8` image arrays, selected action IDs, and resulting image arrays. It receives no positions, task solutions, hidden causal rules or diagnostic state. The evaluator alone calls `diagnostic_state` to count exploration positions, key possession, door opening, and goal completion.

`aura/exploration/validated_progress.py` implements an independent action-conditioned residual predictor over 12 **fixed, label-free image statistics**. Its small matrices are randomly initialized. For each action, the first four observed transitions form a frozen local validation bank (never used for that predictor's gradient updates). Further transitions train only the corresponding action's predictor. It keeps a signed, conservative exponential average of loss reduction, requiring positive improvement across *both* disjoint halves of the fixed bank. It penalizes persistent prediction loss as a *noise/capacity/hidden-state heuristic*, **not** as a proven aleatoric noise measure.

A recurring-pixel-color counter supplies a simple scene-change prior and masks one-off RGB colors; state-action novelty uses the same recurrence-derived scene context. This is an engineered inductive bias and may fail in natural images or visual domains with continuous illumination. The policy also maintains independent action budgets and epsilon exploration. It does not compute exact mutual information or transferable-skill value.

The benchmark `experiments/gridworld_exploration.py` compares three policies under matched interaction budgets. `NoisyTVWorld` adds an **independent sixth action (5)** that changes no physical state or reward, returning an unlearnable 3x3 random pixel patch. The normal **interact action (4)** remains available to operate doors/switches; this removes a confound in an earlier exploratory benchmark that contaminated action 4 itself. Each policy is evaluated on the *same* separate, fixed development transition stream not used in training or local validation. Seeds 0–4 were used for development; seeds 20–24 were subsequently run without retuning. **Neither set is a cryptographically protected verifier holdout.**

## Measured development results (five seeds 0–4, 500 interactions each)

Entries below are means over five independent seeds. `episode-position` counts unique `(episode number, evaluator-only position)` pairs, so they should not be interpreted as distinct global map states. Lower test MSE is better.

| Task | Policy | Episode-position count | Mean held-out feature MSE | Successful episodes |
|---|---|---:|---:|---:|
| key | random | 120.4 | 0.003189 | 0 |
| key | prediction error | 41.0 | 0.003396 | 0 |
| key | validated progress | 85.0 | 0.003516 | 0 |
| switch | random | 122.0 | 0.003119 | 1 |
| switch | prediction error | 40.2 | 0.003394 | 0 |
| switch | validated progress | 86.8 | 0.003434 | 0 |
| noisy TV | random | 118.6 | 0.002674 | 0 |
| noisy TV | prediction error | 41.6 | 0.002960 | 0 |
| noisy TV | validated progress | 104.0 | 0.002814 | 0 |

For the noisy-TV task, the final-half mean fraction selecting its sixth, unproductive action was **15.04% random**, **1.36% prediction-error**, and **1.92% validated-progress**. The prediction-error baseline also avoided the TV; this benchmark **does not** support claiming that raw-error selection universally chases random pixels.

## Fresh-seed confirmation (five seeds 20–24, 500 interactions each)

| Task | Policy | Episode-position count | Mean held-out feature MSE | Successful episodes |
|---|---|---:|---:|---:|
| key | random | 127.2 | 0.003416 | 0 |
| key | prediction error | 48.2 | 0.003574 | 0 |
| key | validated progress | 88.0 | 0.003579 | 0 |
| switch | random | 126.4 | 0.003391 | 0 |
| switch | prediction error | 54.2 | 0.003540 | 0 |
| switch | validated progress | 82.8 | 0.003517 | 0 |
| noisy TV | random | 104.4 | 0.002766 | 0 |
| noisy TV | prediction error | 32.4 | 0.002977 | 0 |
| noisy TV | validated progress | 98.8 | 0.002864 | 0 |

The noise-action fractions in the confirmation sample were **17.44% random**, **1.52% prediction-error**, and **2.08% validated-progress**. The learner avoids spending actions on the unproductive TV but does not outperform random in exploration coverage, prediction error, or task success on this sample.

The implementation's first attempt initially over-selected a **confounded** noisy action (68.2%) because the previous TV wrapper modified the necessary interaction action. This failure triggered an algorithm revision and benchmark repair. Those pre-repair numbers are *not* comparable with the final independent-sixth-action numbers; all three exploration runs must be interpreted with the recorded benchmark version.

## Executable commands

```bash
pytest
python -m experiments.gridworld_exploration --seeds 5 --steps 500 --output runs/gridworld_exploration.json
python scripts/train.py --config configs/validated_curiosity_cpu.yaml --condition curiosity --task key
```

The last command integrates the policy into the real agent, records all observed experiences in its immutable-source ledger and records an inference-backed action-summary. Original ensemble curiosity remains the default. The world model's training updates are independent of this small comparator's own frozen validation bank.

## Further acceptance conditions

- Improve gridworld exploration **and** held-out predictive learning (or control success) over random exploration under independent, preregistered matched-budget trials, rather than merely over one weak baseline.
- Avoid losing necessary interactions while reducing unlearnable novelty; test nonstationary and low-recurrence observation palettes.
- Replace the observation-noise/capacity proxy with uncertainty estimates calibrated for the relevant environment and sensor reliability.
- Demonstrate multi-seed skill acquisition or transfer in the actual gridworld; report intervals and computational overhead.

Failure to satisfy these gates is a research result, not a reason to inflate the section status.
