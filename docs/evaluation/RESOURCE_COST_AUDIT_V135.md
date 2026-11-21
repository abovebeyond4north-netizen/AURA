# AURA-Ω v135 — Public navigation resource-cost audit

**Acceptance status: unchanged.** This is controlled, publicly visible diagnostic evidence.
It is NOT independent evaluation, a fair accounting of the entire autonomous
learning lifecycle, or a reason to promote Section 08/24/30.

## Task traceability

- `AURA-24.3-T2-S1`: partial — environment actions and active probe actions
  are fully accounted in two historical public runs; total training updates
  and development interventions are **not available** from those old records.
- `AURA-24.3-T2-S2`: partial — measured full-episode CPU and wall time
  in a *new* controlled trial batch; total model-selection/search and
  learner-training time not measured.
- `AURA-08.4-T1-S1`: partial — actions, trial CPU, trial wall time, failures
  and goals are reported per policy; training/skill acquisition cost unknown.
- `AURA-24.1-T1-S2`: enhanced — frozen seeds, raw traces, source-controlled
  manifests and SHA-256 output references, but no blind evaluator.

## Reconciled historical results

The records below include actions spent on **failures**. Internal probes are
included in environment actions (do not add them again).

| Study | Controller | Goals | Environment actions | Active probe actions |
|---|---|---:|---:|---:|
| A (216 rows) | EfficientRouteCamera | 20/72 | 860 | 0 |
| A (216 rows) | PersistentPoseCamera | 53/72 | 1978 | 7 |
| A (216 rows) | ComposedRouteCamera | 49/72 | 1204 | 20 |
| B (216 rows) | PersistentPoseCamera | 47/72 | 1938 | 5 |
| B (216 rows) | ComposedRouteCamera | 52/72 | 1269 | 8 |
| B (216 rows) | ConfidenceGatedRouteCamera | 49/72 | 1904 | 5 |

Source-specific public manifest: `docs/evaluation/public_holdout_v1.json`.
Raw study A and B inputs are regenerated from the frozen experiment scripts
and accepted only when their original coverage/summary checks succeed.
Run `python -m tools.navigation_resource_audit --manifest ... --public ...
--gated ... --output ...`.

## New 216-trial public instrumented experiment

`docs/evaluation/public_resource_profile_v1.json` freezes seeds 310101–
310104, 9×9/13×13/15×15 layouts, both movement regimes, and 3%, 15%, 20%
observation noise. Every arm has a 65-action cap. Execution order rotates to
reduce deterministic first-run advantage. Each timed call includes creating
the procedural world, resetting, sensing, choosing actions and episode end.
It does **not** include model training, prior tuning or architecture search.

The first locally controlled run found:

| Controller | Goals | Actions | Measured full-trial CPU seconds | Measured full-trial wall seconds |
|---|---:|---:|---:|---:|
| PersistentPoseCamera | 38/72 | 1644 | 2.036 | 2.037 |
| ComposedRouteCamera | 43/72 | 1007 | 1.778 | 1.779 |
| ConfidenceGatedRouteCamera | 39/72 | 1617 | 2.086 | 2.087 |

Those wall/CPU times are **machine specific**, not invariant benchmark
numbers. Hosted CI will generate its own version-dependent timing records.
Do not compare run totals across machines as controlled differences.
Source files and raw per-trial timings, including failures and counterexamples,
must be retained.

## Commands

```bash
python -m tools.navigation_resource_audit \
  --manifest docs/evaluation/public_holdout_v1.json \
  --public runs/public_holdout_v1.json \
  --gated runs/confidence_gate_v1.json \
  --output runs/navigation_resource_audit_v1.json

python -m tools.navigation_runtime_profile \
  --manifest docs/evaluation/public_resource_profile_v1.json \
  --output runs/resource_profile_public_v1.json

AURA_REQUIRE_PUBLIC_STUDIES=1 python -m pytest -q \
  tests/test_navigation_resource_audit.py \
  tests/test_navigation_runtime_profile.py
```

Both CLIs refuse to overwrite existing evidence. CI uploads JUnit, JSON
exit status, both raw experiment records, the reconciled action budget, and
the new measured full-episode profile even when a job fails.

**Unmeasured costs remain explicit nulls:** model training steps, full
candidate-search costs, extra development interactions, online diagnostics
outside the episode, peak training memory and hardware energy.

## Interpretation

The current public results conflict: Study A favors persistent localization
on goals (53 vs 49), while Study B and the new instrumented batch favor
composition (52 vs 47, and 43 vs 38). None are externally blinded. Four seed
clusters per study and repeated public testing cannot support a
confirmatory generalization or recursive-improvement claim.

The priority following this audit is true learned action-effect calibration
with uncertainty and rollback, followed by frozen independent evaluation.
Protected verifier code, acceptance labels and promotion gates stay intact.
