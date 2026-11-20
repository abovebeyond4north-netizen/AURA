# v60 — Persistent set-valued localization (experimental)

## Mechanism

`PersistentPoseCamera` extends the v57 adaptive survey controller. It retains the baseline motion checks. On ambiguous ordinary mapped movement (no incompatible nonzero vote), a bounded set of possible positions survives across probes. Static palette tiles in successive images and the frozen observed map eliminate conflicting possibilities; absent or dynamic pixels are ignored. Insufficient evidence does not eliminate a hypothesis. Probes are restricted to directions whose targets for every surviving pose are known walkable tiles or walls, ranked by predicted pairwise static-map contrasts. Movement and no-motion alternatives are retained for walkable targets. Wall occupancy is treated as reliable. At most four extra probe actions and twelve active hypotheses (abort when exceeded); all actions count against the existing 65-action limit. A single surviving pose must have at least 12 current static tile matches to the map before commitment. Shared map and position remain unchanged until resolution. If the resolved position differs from the intended route, navigate back via the existing verification checks; nested recovery is disabled while returning. Key pickup, door passage, calibration and nonmovement actions are not modified. This is a hard-constraint set-valued belief model, not a learned or calibrated Bayesian posterior; it relies on the known palette, stationary tile model, reliable map and calibrated action contract.

## Evaluation

Development: seed 90200, layouts 9:6,11:3,13:7,15:4,17:9; 18/30 goals in each arm. Fresh public evaluation: seeds 91100–91101 on 9:4,11:7,13:3,15:8,17:5 and 91200–91201 on 9:2,11:6,13:9,15:5,17:10. Each crosses normal/reversed movement and 0%, 3%, 8% independent random RGB pixel corruption. Matched reset, size, door column and cap (noise streams may diverge as actions diverge). Some geometries overlap earlier studies; seed ranges are new, not independently withheld. Evaluator-only audit checks every attempted `_observe` commit against independently obtained relative true pose before calling the ordinary method. The controller never consumes that audit. The terminal step raises Finished before a final map commitment; no claim is made about unaudited terminal pose or calibration intermediate frames. No comparisons against v50, v58 or independently private evaluation assets in this study.

| Fresh metric | v57 | v60 |
|---|---:|---:|
| Clean |28/40|28/40|
| 3% corruption |16/40|28/40|
| 8% corruption |18/40|24/40|
| Goals |62/120|80/120|
| Raw actions |3664|4581|
| Failure-adjusted actions (65 each failure) |5656|5390|
| Incorrect committed poses recorded |0|0|

18 improved and zero regressed fresh matched outcomes. v60 attempted 144 recoveries across 56 fresh trials, resolved 134, used 151 extra probe actions, and reached two simultaneous hypotheses at most. Forty failures remained: 29 ordinary movement-contract failures, six unresolved persistent beliefs, one ambiguous calibration, four with no exception reason (budget/no success). More raw actions were spent; the fixed failure-penalized total decreased by 266. This is not general learning efficiency or statistical proof of general performance. No default promotion, new Git commit, full regression run or section acceptance change. Section 08 remains partial.

## Verification and reproduction

42 focused tests passed, including persistent multi-step hypotheses, missing-evidence retention, map conflict filtering, safe probe restrictions, evaluator pose audit and unchanged map on unresolved recovery. 240/240 fresh records replayed byte-for-byte with the stronger audit hook. 60 final development records plus 240 fresh records. Protected verifier unchanged from v59. Reproduce with `python -m experiments.persistent_pose_study --seed 91100 --layouts 9:4,11:7,13:3,15:8,17:5 --output out.json`; repeat for the other seeds and layout group. Tests in `verification/section08/v60/tests_command.txt`.

Next: audit the remaining failures by stage (especially calibration and door/goal motion), test v60 on the early-key v58 architecture, and evaluate wider corruption and environment families. Retain v57 and v58 as controls; v50 remains default.
