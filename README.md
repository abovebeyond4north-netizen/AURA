# AURA-Ω — evidence-constrained learning research prototype

**Status: functional bounded prototype; not AGI.** All neural network weights begin from random initialization. An image-based procedural world supplies observations, a CNN+GRU predictive ensemble trains on replay, and an append-only SQLite ledger stores provenance. Controlled modification proposals are independently evaluated in a subprocess, with paired held-out seeds and a conservative promotion decision.

## Install

Recommended Python 3.11+ on Linux/macOS. In a network-enabled environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python scripts/train.py --config configs/cpu_smoke.yaml
python scripts/evaluate.py --config configs/cpu_smoke.yaml
python scripts/test_recursive_learning.py --config configs/cpu_smoke.yaml
```

The offline evaluation workspace used for initial development has PyTorch but no Gymnasium/TensorBoard; `ProceduralWorld` includes a limited Gym-compatible fallback. This fallback is NOT a substitute for the full Gymnasium installation. Install commands cannot be verified offline.

## Learning and limitations

- **Neural training:** randomly initialized CNN observation encoder, image reconstruction, one-step recurrent action-conditioned ensemble and latent variance regularization. Actor Q-values use TD updates on learned embeddings.
- **Exploration:** action-selection scores combine model disagreement, fixed-sample measured learning progress and an empirical image-noise penalty. Scores are heuristics, not guarantees of information gain.
- **Hypotheses:** The ordinary learning loop records observational action effects. The opt-in `investigate` condition runs competing door hypotheses through randomized interventions in the primary gridworld and stores evidence and counterevidence. See [Section 09 evidence and reproduction](docs/section09_acceptance.md).
- **Skills:** repeated action subsequences mined from successful observed episodes; composition is concatenation requiring fresh empirical evaluation. No claim of reliable compositional skill transfer without benchmark evidence.
- **Concepts:** k-means-like latent prototypes accepted only when a held-out supervised proxy improves; no assertion of human-readable semantics.
- **Continual adaptation:** replay-based retention; no experimental result yet proving forgetting reduction.
- **Meta-controller:** cost-sensitive UCB strategy preference (limited to implemented actions). Training does not yet establish that these allocations outperform schedules.
- **Recursive improvement:** typed permitted changes to hyperparameters/dimensions, versioned archive, subprocess with evaluation under paired random seeds. The verifier cannot be configured via candidate modification interface. This is bounded HPO with search adaptation, *not general self-rewriting*.
- **Verifier threat model:** subprocess and restricted configuration isolate accidental candidate changes, **not** hostile same-user OS code. Production security needs container/VM, credentials separation, OS-level quotas, read-only verifier assets and an independent verifier host.

## Verified local evidence

The `docs/verification_report.md` records the original 39-test local suite, five-seed short-run prediction comparison, noisy-TV ablation, continual-learning diagnostic and bounded candidate outcomes. A first candidate met the prototype promotion gate while its successor failed; no two-generation recursive improvement has been shown. `sample_outputs/` contains actual example model weights, SQLite evidence, JSON results and a descriptive figure. **No successful gridworld goal or skill acquisition was observed in the short benchmark.**

## Reproducibility

Outputs in `runs/<config>/` (repeated training creates a unique run subdirectory): `metrics.jsonl`, `checkpoint.pt`, `evidence.sqlite`, optional `development_evaluation.json`, `candidate_lineage.jsonl`. Checkpoints use PyTorch `weights_only=True` on restore. The protected validation seed namespace is separate from training/dev and not passed to the agent; test code is visible to repo developers, so it is not a cryptographic secret. Randomization by environment uses explicit seeds. GPU and CPU numerical identity is not guaranteed.

## Quality bar

All results must be explicitly measured. The smoke test only proves execution; it does not certify useful autonomous learning, transfer, causal discovery or recursive self-improvement. Read `docs/known_limitations.md` before interpreting any benchmark.

## 31-section continuation (incremental implementation)

See [`docs/section_acceptance.md`](docs/section_acceptance.md) for a section-by-section compliance register and explicit remaining acceptance gates. The extension adds:

- `aura/world_model/temporal_heads.py`: direct 1/2/4-step predictors trained during normal episodes; smoke outputs now contain a separate `temporal.pt` checkpoint and held-out horizon errors.
- `aura/skills/online_options.py` and `aura/environments/skill_workbench.py`: empirical context-conditioned options from unlabeled RGB observations and reward, tested for two-stage composition transfer.
- `aura/cognition/intervention_study.py` and `aura/environments/causal_workbench.py`: randomized binary experiments distinguishing confounded observational association from interventional effect.
- `protected_verifier/solvability.py`: evaluator-only breadth-first certification of main gridworld family reachability (not available to the learner).
- `experiments/evaluate_new_mechanisms.py`: reproducible controlled skill/causal benchmark (`python experiments/evaluate_new_mechanisms.py`).

The newer tests and microbenchmarks **do not close the master final acceptance criteria**. Full task mastery, strong continual learning, robust OOD transfer, independent OS isolation, and two successive validated learning-strategy improvements are still unproven.


## Additional controlled verification

Run `python experiments/evaluate_multiscale.py` to reproduce five-seed, equal-budget absolute-vs-residual temporal-head predictions, including action-sequence latent effects. Run `python experiments/evaluate_uncertainty.py` for the frozen-distribution-shift and repeated-outcome-variability diagnostic. The new residual temporal checkpoint schema is v3; the evaluation script detects legacy absolute-image v1 checkpoints and loads them without claiming their untrained latent heads have learned.

The 31-section register continues to mark both **world-model hierarchy** and **uncertainty** partial: synthetic predictive/OOD results have improved, but broad gridworld generalization and independent statistical verification remain unproven.


A further reproducible continual-learning study is available with `python experiments/evaluate_continual.py`: the same randomly initialized world model is trained on an observable contextual rule, then adapted to a second rule with/without matched-budget replay. It measures old-task forgetting, new-task adaptation, dormant latents, rehearsal memory and elapsed time. The full study and 69-test suite are recorded in `docs/verification_report.md`; this is a controlled benchmark, not broad autonomous continual learning.

## Visual learning-progress continuation

Section 8 has an opt-in, unlabeled action-specific learning-progress explorer (`curiosity_policy: validated_progress`), implemented as a small residual predictor with a fixed held-out training-session bank and recurring-color state novelty. It is integrated into agent exploration, and a matched-budget visual gridworld diagnostic is provided:

```bash
python -m experiments.gridworld_exploration --seeds 5 --steps 500
python scripts/train.py --config configs/validated_curiosity_cpu.yaml --condition curiosity --task key
```

The **76-test local suite passes**, but **Section 8 remains incomplete**: under fresh independent development seeds, the policy avoids a separate unproductive TV action and exceeds prediction-error curiosity in observed coverage, yet random exploration still wins on general exploration coverage and common held-out feature prediction. No reliable full-gridworld control success has been demonstrated. The negative comparison is documented without cherry-picking in [`docs/section8_exploration.md`](docs/section8_exploration.md), and the complete 31-section status remains in the acceptance register.

## Section 08 visual-frontier continuation — v4

A new opt-in `visual_frontier` policy learns relative image-camera translations, records observed action transitions and routes back to newly explorable local states, with no access to true simulator positions. It is integrated into the ordinary agent, including external actor/meta-controller actions. The command below runs a bounded CPU smoke test:

```bash
python scripts/train.py --config configs/visual_frontier_cpu.yaml --condition curiosity --task key
python -m experiments.gridworld_exploration --seeds 10 --seed-start 30 --steps 500
python -m experiments.analyze_frontier sample_outputs/visual_frontier_v4_seeds_30_34.json sample_outputs/visual_frontier_v4_seeds_35_39.json
```

**83 original-plus-new regression tests passed before adding the paired-analysis tests; the latest suite has 85 tests.** On ten fresh test seeds per family, inferred frontier navigation visited 34.0/35.2/36.4 distinct physical grid positions on key/switch/TV compared with random's 21.9/23.6/21.6. The 95% paired differences are positive, but this is **not a demonstrated goal-learning or prediction-efficiency advantage**; TV held-out image-feature prediction was worse, and only two successful task episodes occurred across all three families. See [`docs/section8_visual_frontier.md`](docs/section8_visual_frontier.md) for the algorithm, priors, raw data, uncertainty intervals, and negative results. Section 8 remains **partial**.

Verifier v7 applies POSIX per-process kernel resource limits before evaluation; see `docs/verification_report.md` for tested scope and remaining isolation gaps. This does not constitute an arbitrary-code sandbox.

### Protected verifier runtime (v9)

Worker evaluation requires Linux with kernel seccomp support and `libseccomp.so.2` (for Debian/Ubuntu, package `libseccomp2`). The worker fails closed if these controls cannot be installed. This does not replace a separately credentialed filesystem sandbox. Reproduce the regression suite with `python -m pytest -o addopts='' -q`.

### v37 breadth-first learning smoke

```bash
python -m scripts.minimum_architecture --output runs/minimum-v37 --all-tasks --steps-per-task 40
```

The five task runs start independently from random weights. The fixed local development rollout never enters training replay, and its cost is separately budgeted and reported. It is reused for online feedback, not a protected final test. Exit 2 means the two-generation recursive criterion was not met; inspect `report.json` for learning and verifier outcomes. No candidate is deployed.

### v38 experimental reward-route reuse and control evaluation

See `docs/v38_section_progress.md` for all 31 acceptance sections and reproducible evidence. `python -m scripts.cross_section_study --output runs/new_study` performs bounded adaptive and experimental route learning. `python -m scripts.evaluate_control --checkpoint runs/new_study/adaptive/checkpoint.pt --output runs/new_control.json` evaluates the saved greedy neural actor separately from exploration control. The reward-route policy is opt-in (`curiosity_policy: reward_routes`, `--condition curiosity`) and is a fixed-layout development mechanism, not an independently accepted improvement. The protected recursive verifier remains fail-closed.

### Experimental neural behavior distillation (v40)

Given a closed `evidence.sqlite` from a successful reward-route learning run, `python -m experiments.neural_transfer_study --ledger PATH --output NEW_DIRECTORY --seed 41` trains a new recurrent image/action policy from rewarded observed trajectories and reports disjoint-seed closed-loop evaluations, including frozen/blank image ablations. It stores policy weights and raw provenance; it does not modify the original ledger or promote a candidate. See `docs/verification_report.md` for limits and replication results.

### Autonomous shadow neural training (v41)

```bash
python -m scripts.online_imitation_study --output runs/online-imitation-v41
```

Opt-in configuration: `imitation_interval` (new successful episodes per task, zero disables), `imitation_updates`, `imitation_max_candidates`, `imitation_eval_episodes`. At episode boundaries, the scheduler uses verified observed successful trajectories from that task, with at least two distinct seed groups, at most 32 recent eligible episodes and at most 256 transitions per episode. It trains a fresh CPU recurrent policy, runs paired public development rollouts before and after, and stores separate candidate weights and provenance. Gradient updates share the existing optimization budget; evaluation interactions use a separate bounded counter included in total environment interactions. The active explorer and actor never load these candidates. Neither the shadow evaluations nor the scheduler are independent verifier promotion gates. Candidate generation is live-process only, not restart-resumable; the ledger scan is still linear in total history.

### Optional neural inheritance diagnostic (development only)

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m scripts.online_imitation_study --inherit --output runs/lineage
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m experiments.imitation_lineage_study --directory runs/lineage --output runs/lineage/lineage_diagnostic.json
```

This trains a task-local chain of shadow checkpoints on fresh observed demonstration groups. No candidate activates or qualifies for protected promotion. The common-seed report measures behavioral gains and regressions, not proof of autonomous general intelligence.
