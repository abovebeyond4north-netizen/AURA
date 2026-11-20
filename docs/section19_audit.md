# Section 19 · Simulation audit

Added selectable doorway positions and an optional movement reversal triggered by key collection. Existing configurations retain their previous behavior. The public layout generator assigns disjoint doorway columns to the train, development and reserved test splits. The reserved split is reproducible and public.

The existing evaluator-only search found witnesses for 90/90 cases spanning five task families and two board sizes. A fresh environment replayed each witness with pixel corruption and diagnostics disabled. The longest witness took 17 actions against the 65-action limit. Diagnostics remained unavailable through the learner-facing method. The ten focused regression tests passed, and the existing doorway skill retained 100/100 crossings in its original evaluation matrix.

Reproduce the audit with `python -m experiments.environment_audit --output reachability.json`. Witnesses and skill results are in `verification/section19/`.

This audit establishes environment reachability for the enumerated cases. Gymnasium and PyTorch were unavailable in this runtime. Verification with the installed Gymnasium dependency remains open under Section 03.

The opt-in image policy now probes motion after key pickup. Observed landmark displacement selects between normal and reversed controls, followed by a return to the pickup point. In four nontraining geometries, it correctly detected the movement mapping in all 80 trials and passed the doorway in 40/40 reversed-control trials. The fixed controller passed 0/40 with reversal. Both controllers passed 40/40 with normal movement. The records include the extra calibration actions. These are repeat trials over four geometries with noiseless observations, a known calibration trigger, and a two-mapping hypothesis class. They demonstrate local transfer, with official backend verification remaining separate.

Run `python -m experiments.movement_transfer --output movement_evidence`. Raw RGB traces, actions, and scored outcomes are in `verification/section19/`.
