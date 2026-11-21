# AURA-Ω v134 — clustered paired navigation inference

**Status:** public exploratory replication; **not** externally administered,
blind, or sufficient for Section 08, 24, or 25 promotion. No protected
verifier rules or original acceptance statuses were changed.

## Corresponding WBS task IDs

- `AURA-24.3-T1-S1`: seed-cluster bootstrap (implemented; not independently certified)
- `AURA-24.3-T1-S2`: effect sizes and repeated-look limitations (implemented locally)
- `AURA-08.3-T2-S2`: paired discordances and negative outcomes (implemented)
- `AURA-24.1-T2-S2`: fail-closed rejection of missing or duplicated trial evidence (implemented)

All wider tasks remain open pending further evaluation. The work breakdown
is tracked in draft PR #3; this change is proposed separately.

## Audited study results

Controller effects measure **candidate minus baseline goal-success probability**,
using paired episodes with the same seed, layout, movement regime, corruption
and 65-action cap. Each seed contributes one mean to the resampling unit.

| Public study | Candidate vs baseline | Difference | Seed-cluster illustrative 95% bootstrap range | Two-sided sign-flip sensitivity |
|---|---|---:|---:|---:|
| A, 72 pairs | Composed vs persistent pose | −5.56 percentage points | −19.44 to +12.50 pp | 0.750 |
| A, 72 pairs | Efficient route vs persistent pose | −45.83 pp | −59.72 to −36.11 pp | 0.125 |
| B, 72 pairs | Composed vs persistent pose | +6.94 pp | −8.33 to +19.44 pp | 0.500 |
| B, 72 pairs | Confidence-gated vs composed | −4.17 pp | −18.06 to +11.11 pp | 0.625 |

Only **four** seed clusters appear in each public study. All layout,
movement and noise conditions reuse those seeds and are correlated.
The bootstrap exactly enumerates all 4^4 cluster resamples.
The sign-flip result is **a sensitivity diagnostic, not a valid
randomized-trial or multiplicity-corrected p-value**. The controllers
were engineered and repeatedly compared after public outcomes were seen.

Study A has 8 paired wins and 12 paired losses for composition relative to
persistent pose. Study B has 13 paired wins and 8 losses for composition.
Across studies the ranking reverses, so there is no supported claim that
composition reliably generalizes across random seeds and conditions.
Do not pool these adaptively developed public studies into a fictitious
sealed evaluation sample.

## Durability and test gates

`tools/paired_cluster_inference.py` recomputes outcomes from raw paired
records, refuses missing arms, duplicates, malformed successes, corrupted
traces and unbalanced seed-condition clusters, and verifies the public
manifest and raw study summary before emitting its separate report.

Reproduce after collecting raw results:

```bash
python -m tools.public_navigation_study \
  --manifest docs/evaluation/public_holdout_v1.json \
  --output runs/public_holdout_v1.json
python -m experiments.confidence_gate_study \
  --output runs/confidence_gate_v1.json
python -m tools.paired_cluster_inference \
  --manifest docs/evaluation/public_holdout_v1.json \
  --public runs/public_holdout_v1.json \
  --gated runs/confidence_gate_v1.json \
  --output runs/paired_cluster_v1.json
python -m pytest -q tests/test_paired_cluster_inference.py
```

The hosted workflow performs the two-study audit only on Python 3.11
while the general suite runs on 3.11 and 3.12. GitHub Actions uploads
the raw studies, audit, JUnit, exit-code JSON and logs even if tests fail.
A GitHub-hosted workflow is **project-controlled** and cannot satisfy
the independent evaluator gate in issue #1.

## Next work

Pre-register new hidden task families with outside evaluator custody; use
more independent seeds, family-aware splits, frozen controllers and a
per-study alpha-spending/multiple-look policy. Add compute/interaction
accounting for calibration and failed actions, and pair comparisons with
real learned models rather than only engineered navigation rules.
