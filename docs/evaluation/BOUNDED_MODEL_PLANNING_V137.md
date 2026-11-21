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
