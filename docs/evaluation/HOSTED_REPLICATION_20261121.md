# AURA-Ω hosted replication — 2026-11-21

This is a source-commit-bound report of GitHub-hosted, project-controlled tests,
**not** an independently administered external assessment. The current protected
verifier and acceptance register were not modified for this report.

## Reproducibility anchor

- Repository: `abovebeyond4north-netizen/AURA`
- Tested integration commit: `96e3f7ac2796c953ab292c32fc99b3546e8ca3b2`
- Workflow runs: [44895589731](https://github.com/abovebeyond4north-netizen/AURA/actions/runs/44895589731)
  and [44895594097](https://github.com/abovebeyond4north-netizen/AURA/actions/runs/44895594097).
- Hosted Python: 3.11.18 and 3.12.16; full `pytest -ra` exited 0 on both;
  separate `tests/test_gymnasium_conformance.py` exited 0 on both.
- Real installed Gymnasium: 1.2.2 in hosted environment. The test matrix
  includes Gymnasium environment registration, checker and render checks.
- The Linux namespace opt-in suite is excluded by default; successful CI
  must not be interpreted as proof of production sandbox isolation.

## Public controlled Section 08 studies

First fixed public manifest: `docs/evaluation/public_holdout_v1.json`,
SHA-256 `084c199b8cc524fca296b8725fee62ce2d219980cae6ad29a70e5de0f13ca8d2`.
On 72 matched conditions per controller, PersistentPoseCamera succeeded
53 times, ComposedRouteCamera 49, EfficientRouteCamera 20.
Paired composed-versus-persistent: 8 gains, 12 losses, net -4.
The subgroup audit was run on hosted Python 3.11 and refused acceptance promotion.

Second public study: four fixed seeds (220051–220054), three layouts,
two movement regimes, three observation noise conditions; 72 conditions
per controller, 216 total. PersistentPoseCamera 47/72, ComposedRouteCamera
52/72, ConfidenceGatedRouteCamera 49/72. The visible-landmark gate is
therefore not a successful replacement. The hosted Python 3.11 job
reproduced the local success counts and retained the raw output as
`runs/confidence_gate_v1.json`.

These public studies are not statistically independent validation:
repeated seeds, fixed visible conditions, engineered navigation policies
and project-administered execution. The reversal in ranking between the
two studies is evidence of environment sensitivity, not proof of a
reliable universal improvement.

## Section-register reconciliation

The register remains **13/31 locally verified**:
05, 07, 09, 10, 11, 12, 13, 14, 15, 18, 19, 20 and 21.
The 18 other sections are not newly promoted.

- Section 03: real hosted Gymnasium checks and Python 3.11/3.12
  substantiate part of the technology-stack gate, but the complete
  required runtime matrix and deployment requirements are still open.
- Section 08: new subgroup and negative candidate evidence, still partial.
- Section 17: hosted process testing does not establish independent
  isolation of evaluator-private data or an external assessor.
- Sections 25 and 30: no two-generation independent recursive gain
  or full-system certification; remain open.

## Next acceptance work

1. Obtain an externally administered evaluator and immutable receipt
   independent of candidate development (issue #1).
2. Pre-register frozen learned-policy transfer on disjoint generators,
   changed movement and nonoverlapping seeds; report cluster-aware
   uncertainty and all failures.
3. Reproduce integrated runtime recovery and any historical continuation
   merge from actual source bundles before changing acceptance labels.

All observed negative results must remain accessible. This document is
evidence bookkeeping, not a promotion or a new verification gate.
