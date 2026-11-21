# AURA-Ω v137 — Bounded learned-world-model rollout proposals

**Status:** Partial implementation of `AURA-06.3-T2-S1`; no empirical
goal-acquisition benefit demonstrated. Section 06 and Section 22 stay partial.

## What is implemented

`aura.planning.bounded_model_planner.BoundedModelPlanner` takes two RGB
`[3,7,7]` uint8 camera observations (current and an *already observed*
target), a candidate action mask and a trained WorldModel ensemble.

- Normalizes both RGB frames consistently with `WorldModel.train_batch`.
- Enumerates all action sequences of a fixed 1–3-step horizon, subject to
  a maximum evaluation count. Quota exhaustion abstains without running.
- Runs action-conditioned recurrent dynamics heads in inference/no-grad
  mode; scores the final predicted RGB observation against the observed
  target image, plus an ensemble-image disagreement penalty.
- Rejects nonfinite rollouts and optionally abstains when every imagined
  sequence exceeds a specified disagreement limit.
- Preserves training/evaluation mode, network parameters and optimizer
  state. Imagined transitions are **never written** to observed replay,
  the ledger or external reports as facts.
- Refuses untrained models and any target explicitly marked as imagined.

The target-observation source is a caller-supplied label, not an independent
proof of data custody. Learned model predictions may be inaccurate or
overconfident: ensemble image variance is an uncalibrated heuristic.
The planner returns a `PlanProposal` but does **not** execute it, modify a
navigation controller, or claim privileged simulator action legality.

## Verified behavior

The six focused local tests cover:

1. An engineered, fixed image-prediction head selects the candidate best
   matching a pre-observed target image.
2. An untrained model abstains without suggesting an action.
3. Horizon/branch quota exhaustion abstains before model evaluation.
4. Imagined-target and malformed action masks are rejected.
5. A real `WorldModel`, after one genuine optimizer update on a synthetic
   observed-transition batch, yields a finite proposal. Its entire
   `state_dict` is unchanged by planning; training state is restored.
6. Two identical heads have exactly zero disagreement and remain eligible
   under a zero-risk ceiling.

These tests demonstrate component contracts and bounded computation, not
world-model quality or improved control. Source-only tests must not promote
the original acceptance criterion.

## Required before closing Section 06.3-T2

- Train the model using real sampled interactions, not synthetic pixel
  transitions, with disjoint held-out layouts and movement mappings.
- Supply a naturally observed target state without consulting evaluator
  hidden information or collecting target data at evaluation time.
- Compare planner suggestions executed in the environment against
  model-free, fixed and oracle controls with equal interaction and
  **total training/search compute** budgets.
- Include negative trials, changing rules, uncertainty calibration, action
  failure recovery, and independent evaluator custody.
- Preserve immutable manifests, source hashes and statistical corrections
  for repeated development searches.

Protected verifier policies are unchanged. No acceptance or promotion.

## Public offline observed-transition proxy — additional evidence

A reproducible, fixed public experiment is now included at
`experiments/model_plan_proxy_study.py`, with frozen configuration in
`docs/evaluation/model_plan_proxy_v1.json`. It collects **288 actually
observed transitions** from 9×9 worlds (seeds 111–118), trains one model for
80 gradient updates, and evaluates on **120 observed transitions** from 13×13
worlds (disjoint seeds 311–314). No evaluator-side positions or secret
control mappings train the model.

| Measure | Controlled local result |
|---|---:|
| Held-out next-image prediction MSE, before training | 0.174482 |
| Held-out next-image prediction MSE, after training | 0.036087 |
| Offline inverse-action proxy correct | 33 / 120 |
| Always choose action 0 | 30 / 120 |

The proxy is deliberately limited: the *next frame, already obtained from
a real recorded environment step*, is passed as an observed goal image.
Consequently, this is retrospective inverse-action matching and **not**
online goal acquisition, foresight, transfer of a planning policy, or
externally administered evaluation. The 2.5-percentage-point observed
difference against the fixed-action baseline is small and unsupported as a
reliable improvement. All 120 per-trial actions, predictions, correctness
flags and observed-frame SHA-256 hashes are retained; negative matches
must not be dropped.

Reproduce with:

```bash
python -m experiments.model_plan_proxy_study \
  --manifest docs/evaluation/model_plan_proxy_v1.json \
  --output runs/model_plan_proxy_v1.json

AURA_REQUIRE_MODEL_PROXY=1 python -m pytest -q \
  tests/test_bounded_model_planner.py tests/test_model_plan_proxy_study.py
```

The script checks disjoint/unique train/evaluation seeds; the validator rejects
missing, duplicated or altered trials, mismatched summary, invalid action
labels and nonfinite reported prediction losses. It emits non-promoting
`independent=false`, `acceptance_promoted=false` flags and source/manifest
bindings. A 13×13 size change alone is insufficient to claim broad
generalization. There has been one publicly tuned training run without
external confirmation; statistical significance is not established.
