# Section 09 · Hypothesis and causal discovery

**Status: verified locally for the Section 09 gridworld acceptance criterion.** AURA identified when key collection or switch activation controlled a door in 24 of 24 stable studies. It rejected every candidate in six studies with changing mechanisms and withheld judgment in six studies with an unreadable camera. The updated project register lists five locally verified sections out of 31.

## What the agent did

The `investigate` condition of the normal training command runs the existing hypothesis engine in the primary, partially observed gridworld. It independently varies key collection and switch activation. It reaches the door, performs the same final interaction across conditions, and checks both its appearance and physical traversability. The camera protocol knows the public tile palette and cardinal action interface. The agent receives no mechanism label, simulator state, or evaluator solution.

The engine constructs four candidate response tables over these exposures: key, switch, both required, and either sufficient. A fifth unconstrained response model is used when choosing experiments. Bayesian updates track evidence and counterevidence. The discovery policy samples interventions by their expected reduction in uncertainty, with a nonzero exploration probability for every arm. Confirmation uses a fresh balanced random order fixed before the first observation. A candidate must meet the fixed response bands in all four arms.

| Section requirement | Implementation and executed check |
|---|---|
| Generate and maintain competitors | Four structured `Hypothesis` records contain expressions, assumptions, contexts, confidence, proposed tests, and evidence/counterevidence IDs. Successive revisions are written to semantic memory. |
| Select experiments by discriminatory value | When the two exposures always co-occurred in the clean controls, all four response rules had identical likelihood. The policy then assigned higher probability to key-only and switch-only interventions. Revision and changing allocation are regression tested. |
| Measure interventions in the primary environment | The policy steers from RGB frames, executes the assigned interventions, and checks attempted passage at the door. The evaluator independently checked both preparation states and the true door state for every measured trial. |
| Distinguish correlation from intervention | In a separate latent-confounding control, passive associations ranged from 0.790 to 0.823 across six seeds. Each randomized study supported a negligible effect for the confounded null and a positive effect for the effective mechanism. |
| Revise, abstain, and allow all candidates to fail | The mixed hidden-rule family rejected all four fixed explanations. Unreadable sensors and short/truncated trials remained unresolved. Missing outcomes widen the reported bounds. |
| Preserve provenance | Design, assignment, real RGB/action sequences, derived measurements, and hypotheses are recorded separately. All 11,616 traces replayed; all 36 SQLite hash chains verified. |

## Results

The original and fresh-seed replication used three paired design seeds apiece for each case. Every cell in the table reports all six studies. “Door readings” counts trials in all phases for which the camera protocol could verify passage or blockage. Each such reading matched evaluator-only state.

| Primary gridworld and ground truth | Correct study decisions | Verified door readings / attempted trials |
|---|---:|---:|
| 9×9, key | 6/6 key | 1,680/1,680 |
| 9×9, switch | 6/6 switch | 1,680/1,680 |
| 11×11, shifted door and 1% pixel corruption, key | 6/6 key | 1,657/1,680 |
| 11×11, shifted door and 1% pixel corruption, switch | 6/6 switch | 1,664/1,680 |
| 9×9, hidden rule sampled on each reset | 6/6 all candidates rejected | 3,216/3,216 |
| 9×9, masked camera | 6/6 inconclusive | 0/1,680 |
| **Total** | **36/36** | **9,897/11,616** |

The study executed 123,029 primitive actions. Each run used eight coupled trials, 16 adaptive discovery trials, then 64 confirmation trials per intervention (128 per intervention for the mixed population). The four exact binomial intervals use a Bonferroni correction at 5% familywise error **per study**, conditional on independent resets from the declared population. Each lower bound treats missing responses as closed; each upper bound treats them as open. The decision requires the whole interval to fall in the predicted low (≤0.25) or high (≥0.75) response band. Public simulation seeds are paired across case variants. Individual episodes and runs do not represent independent environments sampled from the real world.

The first attempted suite stopped on a shifted key-controlled door. Its readings were correct when available, but along a nearly uniform wall the camera lost motion evidence too often; 11 of 64 both-treatment confirmation trials were unreadable. The revised survey prefers an interior viewpoint when new coverage per action ties and can reobserve initial preparation landmarks from a stationary wall bump. The original protocol (Git commit `4789a9d`) is preserved in Git. The failed run is recorded in `verification/section09/initial_*`. The implementation and fresh replication seeds were committed together (`c8f4907`) before running the final checks. Model classes, budgets, response bands, and ground-truth criteria were the same for both final suites.

## Reproduction

Run from the source directory with Python, NumPy and SciPy available. The primary environment supports the repository's offline API fallback. The local acceptance runs used that fallback. For a direct investigation, choose a new output directory (one ledger per investigation):

```sh
python scripts/train.py --condition investigate --task switch --study-seed 91 --output runs/door-investigation
python -m unittest discover -s tests -p test_door_study.py -v
python -m experiments.section09_acceptance --output runs/section09-original
python -m experiments.section09_acceptance --replicate --output runs/section09-replication
```

The package contains both sets of SQLite trial ledgers under `evidence/`, the two full JSON reports under `verification/section09/` in the source tree, and the earlier failed shifted-layout ledger. To inspect a single trial, follow its `door_measurement` record to its assignment and observation. The `replay` function reconstructs the decision from the recorded images and verifies every requested action.

The five focused unittest methods passed, including five existing Section 09 regression checks invoked by one method. Python compilation passed. The full neural pytest suite was unavailable in this runtime. This result establishes the bounded Section 09 mechanism-identification criterion. It assumes a known camera palette, known intervention primitives, and valid physical manipulations. Free-form causal theories, arbitrary worlds, neural acquisition of the probe protocol, and protected candidate promotion are outside the demonstrated scope.
