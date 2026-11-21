# AURA-Ω v136 — Observation-only motion confidence and rollback

**Status:** Experimental, public and controlled. Section 08 remains partial;
Section 17 protected rules unchanged; no claim of learned policy improvement
or independent generalization.

## P0 traceability

- `AURA-08.2-T2-S1` — Partially implemented: non-oracle visual
  raw-action-effect counts, empirical support and per-action confidence;
  contradictory visual evidence invalidates the confidence regime.
- `AURA-08.2-T2-S2` — Controlled checks: no evaluator position or
  simulator action mapping enters the confidence learner. The comparative
  experiment separately retains privileged diagnostic traces for the auditor.
- `AURA-22.4-T1-S1` — In-memory transactional rollback of a failed
  reidentification commit was tested. This is **not** disk checkpoint,
  process-crash recovery, or reversal of physical world movement.
- `AURA-24.1-T2-S2` — Public changed-action negative outcomes preserved.

## Mechanics and limitations

`OnlineMotionConfidence` consumes only two consecutive uint8 RGB frames
and a raw action identifier. A transition counts only if at least two distinct,
fixed visible landmarks ('crate', 'switch', 'goal') agree on one cardinal
camera displacement. Unknown, occluded, changing, blocked, conflicting and
unsupported movements cause abstention; keys and doors are excluded because
they can disappear or change. The independent observations are categorical
counts, **not a calibrated statistical posterior**.

A complete raw-action mapping is committed to this *advisory* confidence
ledger only when every raw control has at least six verified observations,
a Dirichlet(1) descriptive confidence above 0.65 and all four inferred
directions are distinct. A contradiction clears confidence evidence,
records a rollback and requires relearning; it never silently rewrites a
trusted operational mapping. Existing `EvidenceChangeCamera` alone
controls physical navigation and mapping reidentification. The new class
also restores mapping, position, image, map, landmarks and resolution
counts on exceptions during reidentification commit.

Rollback cannot undo physical environment movement and is not an
isolated security boundary.

## Frozen public experiment

Manifest: `docs/evaluation/motion_confidence_public_v1.json`
SHA-256: `a62b47f3d74bc46c0daf8214436a17f4d737da71f4abfcfb003d39fe583101f5`

Two public seeds, 9×9 and 13×13 layouts, two prior movement mappings,
0% and 8% pixel noise, and three change regimes (none, after door, late)
yield 48 paired configurations (96 records). Maximum 65 actions per
controller per trial, including failed trials and internal probes.

| Controller | Goals reached | Total actions | Pose/mapping error trials |
|---|---:|---:|---:|
| EvidenceChangeCamera | 40 / 48 | 1,180 | 0 / 0 |
| ConfidenceRollbackCamera | 40 / 48 | 1,180 | 0 / 0 |

Every controller had eight unsuccessful trials, all reporting
`reidentified pose cannot return to intended route`.

The new advisory confidence tracker recorded 68 usable raw-action
observations, 636 abstentions and 15 confidence-regime rollbacks, but
**no trial collected enough evidence to commit a full bijective action
mapping**. The result establishes no gain in goal success or efficiency.
The existing recovery algorithm determines the actual actions.

## Reproduction and acceptance limitations

```bash
python -m pytest -q tests/test_motion_confidence.py \
  tests/test_motion_confidence_study.py \
  tests/test_change_dynamics.py tests/test_learned_dynamics.py

python -m experiments.motion_confidence_study \
  --manifest docs/evaluation/motion_confidence_public_v1.json \
  --output runs/motion_confidence_public_v1.json

AURA_REQUIRE_MOTION_STUDY=1 python -m pytest -q \
  tests/test_motion_confidence.py tests/test_motion_confidence_study.py
```

The experiment fails on missing, duplicate or unexpected paired inputs,
invalid actions and inconsistent summary, and refuses to overwrite the
original report. The CI record binds the source commit and manifest hash.
The local focused test suite passed 21 tests; one test deliberately skips
until the generated public study dataset is present.

**Not complete:** learned movement policy selection, statistically
calibrated transition confidence, changed-dynamics generalization
over independent tasks, multi-environment transfer, or outside-evaluator
reproduction. Do not promote any original section based on these data.
