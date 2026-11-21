# Section 08 — Public replication and failure boundaries

**Status: Partial.** These are controlled, public, engineered-controller experiments.
No learned strategy selection, blind generalization certification, or externally
administered independent validation is claimed.

## Experiment A — v131, frozen public manifest

Manifest: `docs/evaluation/public_holdout_v1.json`, SHA-256
`084c199b8cc524fca296b8725fee62ce2d219980cae6ad29a70e5de0f13ca8d2`.
The 216 paired trials (72 per controller) were repeated exactly on hosted CI.

| Controller | Successes | Rate |
|---|---:|---:|
| EfficientRouteCamera | 20/72 | 27.78% |
| PersistentPoseCamera | 53/72 | 73.61% |
| ComposedRouteCamera | 49/72 | 68.06% |

Composition versus persistent: eight pairwise improvements, 12 pairwise
regressions. Seed-level paired net improvements: 210041 -1, 210042 -3,
210043 -4, 210044 +4.

Exploratory stratification identifies the greatest losses in the 15×15 layout
(persistent 19/24 versus composed 13/24) and at 20% pixel corruption
(persistent 12/24 versus composed 7/24). These are overlapping conditions;
do not treat the strata as independent experiments. The repeated seed set is
small and public, so aggregate percentages cannot establish transfer.

Reproduction:
```bash
python -m tools.public_navigation_study --manifest docs/evaluation/public_holdout_v1.json --output runs/public_holdout_v1.json
python -m tools.navigation_audit --manifest docs/evaluation/public_holdout_v1.json --results runs/public_holdout_v1.json --output runs/navigation_audit_v1.json
```

## Experiment B — Landmark-gated engineered candidate

A separate predeclared public matrix uses seeds 220051–220054, layouts
9×9/13×13/15×15, two altered-movement regimes and 3%, 15%, 20% corruption.
Every policy receives the same 65-action cap. The gate uses only landmarks
in the initial egocentric RGB observation, never evaluator-only state. It
selects the early-key variant if both key and door were visible; otherwise
uses the original persistent-pose preparation.

Local fallback results from 72 trials per controller:

| Controller | Successes | Actions, including failures |
|---|---:|---:|
| PersistentPoseCamera | 47/72 | 1,938 |
| ComposedRouteCamera | **52/72** | 1,269 |
| ConfidenceGatedRouteCamera | 49/72 | 1,904 |

The new gate did **not** outperform composition; it remains an experimental,
nondefault policy. The ranking of persistent and composed differs from
Experiment A. Every raw success, failure, trace and localization audit must
be retained. Hosted source-commit-bound reproduction is required to
supplement these local results.

Reproduction:
```bash
python -m experiments.confidence_gate_study --output runs/confidence_gate_v1.json
```

## Advancement gate

Do not promote Section 08 on engineered public controller comparisons.
Required next: non-oracle representation/strategy learning, preregistered
multi-task transfer with frozen policy and independent environment generator,
statistical uncertainty accounting for seed clusters, and a genuinely
external assessor following `EXTERNAL_EVALUATOR_PROTOCOL.md` (issue #1).

Protected verifier files and acceptance status are unchanged.
