# Section 08 — v53 probabilistic pose-belief experiment

Status: experimental, **not promoted**. The validated v50 navigation controller remains the operational baseline. Overall acceptance remains 13/31 locally verified; Section 08 is partial.

## Change

Added `aura/exploration/pose_belief.py`: per-transition five-hypothesis displacement posterior over stationary pixels. It uses a declared independent RGB corruption model, a motion prior, and a strict confidence threshold. Ambiguous observations return `None`; no inferred pose is committed. The module is not connected to the active controller.

## Verification

`python -m pytest -q -o addopts='' tests/test_pose_belief.py tests/test_active_goal.py tests/test_passage_goal.py`

Result: **11 passed** in the working environment. This is a focused unit/regression test, not a matched-budget navigation experiment. No success-rate improvement or general localization capability is claimed.

## Limitations and next gates

- The likelihood model assumes approximately independent pixel corruption, not occlusion, camera rotation, or correlated noise.
- The prior is engineered; model parameters are not learned from data.
- Shift hypotheses are evaluated independently per transition. Persistent pose tracking, map fusion, and active sensing remain future work.
- Run matched 11x11 and unseen-layout goal acquisition trials at 0%, 3%, and higher corruption, then compare against v50 with frozen seeds and identical action budgets.
- Run full regression tests before promotion. Keep the protected verifier unchanged.
