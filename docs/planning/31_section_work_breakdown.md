# AURA-Ω: 31-section hierarchical execution backlog

> **Planning artifact; not an acceptance certificate.** Anchored to the original 31 entries in `docs/section_acceptance.md` and the AURA-Ω architecture specification. Existing acceptance status is carried forward unchanged. Every checklist item is **unassessed** by this planning exercise, including tasks associated with locally verified sections. Review existing evidence before changing an individual checkbox.

## Snapshot and operating rules

- Scope: **31 sections → 100 subsections → 200 tasks → 400 concrete subtasks**.
- Acceptance register: **13 locally verified, 14 partial, 3 open, 1 ongoing**. Local verification is not general or independent certification.
- P0 = highest-impact blocking work; P1 = integrated development/replication; P2 = maintenance and expanded independent replication.
- Linked sections are cross-references, **not a strict dependency graph**. Some gates are mutually dependent and must be advanced incrementally.
- **Evidence rule:** for each completed leaf, record source commit SHA, exact command, dependency versions, trial manifest/seed ownership, success and failure traces, exit code, artifact digest, and limitation statement.
- **Promotion rule:** no unchecked evidence is treated as successful. Do not modify protected verifier policy or the 31 original criteria in order to pass a test.
- **Independent rule:** project-controlled GitHub Actions are official platform evidence, **not externally administered independent evaluation**.
- **Execution rule:** if a gate is blocked (external assessor, kernel capability, unavailable platform), log the blocker, keep it open, and advance the next reachable P0 dependency.
- **Completion rule:** stop the hourly completion workflow only once all 31 are individually supported by their original acceptance requirements and a full audit is recorded.

## Quick index

| Section | Original heading | Carried status | Priority | Subsections | Tasks | Subtasks |
|---|---|---|---|---:|---:|---:|
| [01](#section-01) | Mission | Open | P0 | 3 | 6 | 12 |
| [02](#section-02) | Non-negotiable engineering requirements | Partial | P0 | 3 | 6 | 12 |
| [03](#section-03) | Technology stack | Partial | P0 | 3 | 6 | 12 |
| [04](#section-04) | System architecture | Partial | P1 | 3 | 6 | 12 |
| [05](#section-05) | Observation & representation learning | Verified locally | P2 | 3 | 6 | 12 |
| [06](#section-06) | Hierarchical world model | Partial | P0 | 3 | 6 | 12 |
| [07](#section-07) | Uncertainty | Verified locally | P1 | 3 | 6 | 12 |
| [08](#section-08) | Curiosity | Partial | P0 | 4 | 8 | 16 |
| [09](#section-09) | Hypothesis and causal discovery | Verified locally | P1 | 3 | 6 | 12 |
| [10](#section-10) | Evidence & memory | Verified locally | P2 | 3 | 6 | 12 |
| [11](#section-11) | Concept discovery | Verified locally | P2 | 3 | 6 | 12 |
| [12](#section-12) | Skills | Verified locally | P1 | 3 | 6 | 12 |
| [13](#section-13) | Curriculum | Verified locally | P1 | 3 | 6 | 12 |
| [14](#section-14) | Continual learning | Verified locally | P1 | 3 | 6 | 12 |
| [15](#section-15) | Meta-learning | Verified locally | P1 | 3 | 6 | 12 |
| [16](#section-16) | Recursive self-improvement | Partial | P0 | 4 | 8 | 16 |
| [17](#section-17) | Independent verifier | Partial | P0 | 4 | 8 | 16 |
| [18](#section-18) | Value of computation | Verified locally | P1 | 3 | 6 | 12 |
| [19](#section-19) | Simulation environment | Verified locally | P1 | 3 | 6 | 12 |
| [20](#section-20) | Repository structure | Verified locally | P2 | 3 | 6 | 12 |
| [21](#section-21) | Shared typed interfaces | Verified locally | P2 | 3 | 6 | 12 |
| [22](#section-22) | Core integrated loop | Partial | P0 | 4 | 8 | 16 |
| [23](#section-23) | Training objectives | Partial | P1 | 3 | 6 | 12 |
| [24](#section-24) | Benchmarks | Partial | P0 | 4 | 8 | 16 |
| [25](#section-25) | Recursive validation | Open | P0 | 4 | 8 | 16 |
| [26](#section-26) | Execution phases | Partial | P1 | 3 | 6 | 12 |
| [27](#section-27) | Commands and artifacts | Partial | P1 | 3 | 6 | 12 |
| [28](#section-28) | Quality and tests | Partial | P0 | 4 | 8 | 16 |
| [29](#section-29) | Repository workflow | Partial | P1 | 3 | 6 | 12 |
| [30](#section-30) | Final acceptance | Open | P0 | 3 | 6 | 12 |
| [31](#section-31) | Engineering principle | Ongoing | P1 | 3 | 6 | 12 |

## Suggested execution sequence

Prioritize **P0 parallel tracks**: (A) Sections 03 + 28 for durable official dependency/platform checks; (B) Sections 06 + 08 + 24 for genuine held-out learning and generalization; (C) Sections 02 + 17 + 16 + 25 for verifier isolation and successive independent improvements. Sections 22 and 30 then integrate and audit the results. Existing locally verified gates stay evidence-linked and are not reopened merely because this is a new backlog.

## Section-by-section breakdown

<a id="section-01"></a>
### Section 01 — Mission

**Carried status:** Open · **Priority:** P0 · **Related gates:** 02, 06, 08, 16, 17, 24, 25, 30

#### AURA-01.1 — Operational definition of success

- [ ] **AURA-01.1-T1 — Define measurable autonomy targets**
  - [ ] `AURA-01.1-T1-S1` Specify predictive, planning, adaptation and recovery outcomes per task family
  - [ ] `AURA-01.1-T1-S2` Fix minimum gains, maximum cost and permitted failure rates before testing
- [ ] **AURA-01.1-T2 — Establish baseline comparisons**
  - [ ] `AURA-01.1-T2-S1` Run random, fixed-curriculum, model-free and bounded model-based baselines
  - [ ] `AURA-01.1-T2-S2` Match interaction counts, compute, seeds and starting weights across variants

#### AURA-01.2 — Generality and self-teaching

- [ ] **AURA-01.2-T1 — Test skills beyond a single gridworld**
  - [ ] `AURA-01.2-T1-S1` Add unfamiliar task mechanics and non-gridworld observation families
  - [ ] `AURA-01.2-T1-S2` Separate adaptation after interaction from memorized environment-specific rules
- [ ] **AURA-01.2-T2 — Demonstrate self-directed learning**
  - [ ] `AURA-01.2-T2-S1` Log goals chosen without external task labels and their rationale
  - [ ] `AURA-01.2-T2-S2` Compare goal choices against fixed and random exploration schedules

#### AURA-01.3 — Mission closure and limitations

- [ ] **AURA-01.3-T1 — Build mission-level evidence ledger**
  - [ ] `AURA-01.3-T1-S1` Link every capability claim to code, immutable trial results and evaluation version
  - [ ] `AURA-01.3-T1-S2` Record counterexamples, regressions and explicit out-of-scope claims
- [ ] **AURA-01.3-T2 — Gate overall mission declaration**
  - [ ] `AURA-01.3-T2-S1` Require independent replication and successful two-generation improvement
  - [ ] `AURA-01.3-T2-S2` Keep mission Open until dependent original sections satisfy their own gates

**Section review gate:** Reconcile the above evidence with Section 01 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-02"></a>
### Section 02 — Non-negotiable engineering requirements

**Carried status:** Partial · **Priority:** P0 · **Related gates:** 17, 21, 28

#### AURA-02.1 — Initialization and observable learning

- [ ] **AURA-02.1-T1 — Audit learning-from-scratch initialization**
  - [ ] `AURA-02.1-T1-S1` Inspect checkpoint loading and eliminate unintended pretrained weights
  - [ ] `AURA-02.1-T1-S2` Test two fresh initializations for independence and deterministic seeded behavior
- [ ] **AURA-02.1-T2 — Enforce evidence provenance**
  - [ ] `AURA-02.1-T2-S1` Label observed, simulated, inferred and externally supplied records distinctly
  - [ ] `AURA-02.1-T2-S2` Reject laundering imagined data into observed replay or acceptance ledgers

#### AURA-02.2 — Isolation and protected verification

- [ ] **AURA-02.2-T1 — Harden execution boundary**
  - [ ] `AURA-02.2-T1-S1` Reproduce namespace, read-only mounts, seccomp and descriptor-closure probes
  - [ ] `AURA-02.2-T1-S2` Run real worker through production isolation rather than only trusted microprobes
- [ ] **AURA-02.2-T2 — Keep verifier protections immutable**
  - [ ] `AURA-02.2-T2-S1` Hash trusted policy, threshold and evaluator code before and after runs
  - [ ] `AURA-02.2-T2-S2` Attempt candidate edits to protected files, configurations and reward definitions; require rejection

#### AURA-02.3 — Reproducibility and failure handling

- [ ] **AURA-02.3-T1 — Specify compute and resource limits**
  - [ ] `AURA-02.3-T1-S1` Set CPU, memory, file, process, wall-time and network ceilings
  - [ ] `AURA-02.3-T1-S2` Verify timeout, quota and denial outcomes are captured without orphaned processes
- [ ] **AURA-02.3-T2 — Validate failure-closed operation**
  - [ ] `AURA-02.3-T2-S1` Inject missing dependencies, malformed reports and ledger corruption
  - [ ] `AURA-02.3-T2-S2` Ensure failed prerequisites produce nonzero exits and no silent fallback promotion

**Section review gate:** Reconcile the above evidence with Section 02 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-03"></a>
### Section 03 — Technology stack

**Carried status:** Partial · **Priority:** P0 · **Related gates:** 20, 28

#### AURA-03.1 — Supported dependency matrix

- [ ] **AURA-03.1-T1 — Freeze supported runtimes**
  - [ ] `AURA-03.1-T1-S1` Pin Python, PyTorch, NumPy, Gymnasium and test-tool versions in lock/constraints
  - [ ] `AURA-03.1-T1-S2` Execute fresh installs under Python 3.11 and 3.12 on hosted Linux
- [ ] **AURA-03.1-T2 — Validate core numerical backends**
  - [ ] `AURA-03.1-T2-S1` Test CPU-only training and deterministic inference
  - [ ] `AURA-03.1-T2-S2` Exercise optional GPU/MPS backends or explicitly document unavailable devices

#### AURA-03.2 — Official simulator and telemetry interfaces

- [ ] **AURA-03.2-T1 — Certify Gymnasium API**
  - [ ] `AURA-03.2-T1-S1` Run official checker for all task families including render and seed semantics
  - [ ] `AURA-03.2-T1-S2` Test gym.make registration, step/reset spaces, truncation and close lifecycle
- [ ] **AURA-03.2-T2 — Verify experiment telemetry**
  - [ ] `AURA-03.2-T2-S1` Confirm TensorBoard/scalar output, run IDs and checkpoint references
  - [ ] `AURA-03.2-T2-S2` Rerun logging-disabled mode without breaking model training

#### AURA-03.3 — Portability and dependency resilience

- [ ] **AURA-03.3-T1 — Test clean-machine setup**
  - [ ] `AURA-03.3-T1-S1` Run editable install and commands in an isolated environment
  - [ ] `AURA-03.3-T1-S2` Verify documented entrypoints without hidden local paths or untracked datasets
- [ ] **AURA-03.3-T2 — Track unsupported platforms**
  - [ ] `AURA-03.3-T2-S1` Probe macOS/Linux and optional isolation features separately
  - [ ] `AURA-03.3-T2-S2` Mark unsupported combinations explicitly rather than treating skips as passes

**Section review gate:** Reconcile the above evidence with Section 03 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-04"></a>
### Section 04 — System architecture

**Carried status:** Partial · **Priority:** P1 · **Related gates:** 20, 21, 22

#### AURA-04.1 — Component contracts

- [ ] **AURA-04.1-T1 — Review all architectural layers**
  - [ ] `AURA-04.1-T1-S1` Map perception, world model, policy, memory, skills, curriculum and verifier to modules
  - [ ] `AURA-04.1-T1-S2` Identify missing wiring and mocked/simplified substitutes
- [ ] **AURA-04.1-T2 — Enforce directional dependencies**
  - [ ] `AURA-04.1-T2-S1` Keep verifier and evaluator outside candidate-controlled imports
  - [ ] `AURA-04.1-T2-S2` Add boundary tests preventing runtime components from mutating trusted assets

#### AURA-04.2 — End-to-end orchestration

- [ ] **AURA-04.2-T1 — Validate startup-to-checkpoint flow**
  - [ ] `AURA-04.2-T1-S1` Trace one fresh episode through observation, action, training, memory and save
  - [ ] `AURA-04.2-T1-S2` Check identities, seeds and version tags across the full trajectory
- [ ] **AURA-04.2-T2 — Exercise multi-component adaptation**
  - [ ] `AURA-04.2-T2-S1` Run a task that requires a learned model plus retrieved skill and updated curriculum
  - [ ] `AURA-04.2-T2-S2` Prove integration improves an outcome relative to components individually

#### AURA-04.3 — Reliability and extensibility

- [ ] **AURA-04.3-T1 — Introduce fault-isolation contracts**
  - [ ] `AURA-04.3-T1-S1` Define recovery for stalled environments, invalid observations and model divergence
  - [ ] `AURA-04.3-T1-S2` Verify no partial state is marked committed after exceptions
- [ ] **AURA-04.3-T2 — Validate pluggable environments**
  - [ ] `AURA-04.3-T2-S1` Replace the gridworld via typed environment adapter
  - [ ] `AURA-04.3-T2-S2` Confirm agent logic remains environment-agnostic without special-case oracle state

**Section review gate:** Reconcile the above evidence with Section 04 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-05"></a>
### Section 05 — Observation & representation learning

**Carried status:** Verified locally · **Priority:** P2 · **Related gates:** 03, 21

#### AURA-05.1 — Reproduce existing local gate

- [ ] **AURA-05.1-T1 — Re-run encoder learning regression**
  - [ ] `AURA-05.1-T1-S1` Confirm parameter updates and held-out loss reduction from random initialization
  - [ ] `AURA-05.1-T1-S2` Preserve five-seed results, deviations and failed runs
- [ ] **AURA-05.1-T2 — Inspect representation collapse**
  - [ ] `AURA-05.1-T2-S1` Measure latent per-dimension variance and dead-feature count
  - [ ] `AURA-05.1-T2-S2` Abort or alert on collapsed encodings while retaining traces

#### AURA-05.2 — Robust visual representations

- [ ] **AURA-05.2-T1 — Expand observation distributions**
  - [ ] `AURA-05.2-T1-S1` Include occlusion, color perturbation, distractors and spatial scaling
  - [ ] `AURA-05.2-T1-S2` Evaluate each transformation with held-out seeds and matched data budget
- [ ] **AURA-05.2-T2 — Separate observation from oracle state**
  - [ ] `AURA-05.2-T2-S1` Restrict training features to camera/allowed sensor channels
  - [ ] `AURA-05.2-T2-S2` Test that hidden grid coordinates are unavailable to learned policy

#### AURA-05.3 — Cross-task transfer

- [ ] **AURA-05.3-T1 — Measure frozen encoder reuse**
  - [ ] `AURA-05.3-T1-S1` Transfer representations to unseen task mechanics without encoder retraining
  - [ ] `AURA-05.3-T1-S2` Compare against fresh random and pretrained-on-source baselines
- [ ] **AURA-05.3-T2 — Document limits and uncertainty**
  - [ ] `AURA-05.3-T2-S1` Bootstrap performance by held-out environment cluster
  - [ ] `AURA-05.3-T2-S2` Record task families where latent transfer does not improve performance

**Section review gate:** Reconcile the above evidence with Section 05 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-06"></a>
### Section 06 — Hierarchical world model

**Carried status:** Partial · **Priority:** P0 · **Related gates:** 05, 07, 19, 23

#### AURA-06.1 — Temporal and multiscale predictions

- [ ] **AURA-06.1-T1 — Extend horizons safely**
  - [ ] `AURA-06.1-T1-S1` Train and test 1/2/4/8-step action-conditioned targets
  - [ ] `AURA-06.1-T1-S2` Compare rollout accuracy, drift and equal-update baselines across seeds
- [ ] **AURA-06.1-T2 — Validate scale hierarchy**
  - [ ] `AURA-06.1-T2-S1` Connect coarse dynamics and fine visual residual heads
  - [ ] `AURA-06.1-T2-S2` Ablate each hierarchy level on held-out layouts and dynamics

#### AURA-06.2 — State and uncertainty fidelity

- [ ] **AURA-06.2-T1 — Improve predictive calibration**
  - [ ] `AURA-06.2-T1-S1` Calibrate ensemble disagreement versus actual per-horizon errors
  - [ ] `AURA-06.2-T1-S2` Detect nonstationary action effects and issue uncertainty flags
- [ ] **AURA-06.2-T2 — Preserve checkpoint compatibility**
  - [ ] `AURA-06.2-T2-S1` Test v1/v2/v3 checkpoint upgrades and corrupted payload rejection
  - [ ] `AURA-06.2-T2-S2` Roundtrip model, optimizer, RNG and dynamics-schema state after restart

#### AURA-06.3 — Transfer and operational usefulness

- [ ] **AURA-06.3-T1 — Evaluate changed movement**
  - [ ] `AURA-06.3-T1-S1` Train a dynamics model on one action mapping and adapt on a held-out mapping
  - [ ] `AURA-06.3-T1-S2` Compare adaptation cost with fixed-map and oracle-map controls
- [ ] **AURA-06.3-T2 — Prove planning benefit**
  - [ ] `AURA-06.3-T2-S1` Feed learned rollouts into a bounded planner without privileged state
  - [ ] `AURA-06.3-T2-S2` Measure goal successes and compute relative to model-free/fixed heuristics

**Section review gate:** Reconcile the above evidence with Section 06 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-07"></a>
### Section 07 — Uncertainty

**Carried status:** Verified locally · **Priority:** P1 · **Related gates:** 06, 19, 24

#### AURA-07.1 — Types of uncertainty

- [ ] **AURA-07.1-T1 — Separate epistemic from observation noise**
  - [ ] `AURA-07.1-T1-S1` Estimate ensemble variance independently of pixel corruption
  - [ ] `AURA-07.1-T1-S2` Verify uncontrollable-TV noise is not mistaken for learnable novelty
- [ ] **AURA-07.1-T2 — Reproduce conditional-risk local gate**
  - [ ] `AURA-07.1-T2-S1` Replay nominal and shifted transition regimes
  - [ ] `AURA-07.1-T2-S2` Validate frozen risk limits and false-alert rates

#### AURA-07.2 — Calibration and decision integration

- [ ] **AURA-07.2-T1 — Assess probability calibration**
  - [ ] `AURA-07.2-T1-S1` Build reliability curves, Brier scores and coverage by task family
  - [ ] `AURA-07.2-T1-S2` Stratify error by rare transitions and nonstationarity
- [ ] **AURA-07.2-T2 — Integrate conservative action selection**
  - [ ] `AURA-07.2-T2-S1` Stop/recalibrate when state uncertainty exceeds a preregistered limit
  - [ ] `AURA-07.2-T2-S2` Measure avoided errors and extra action costs against ungated controls

#### AURA-07.3 — Out-of-distribution monitoring

- [ ] **AURA-07.3-T1 — Challenge unknown dynamics**
  - [ ] `AURA-07.3-T1-S1` Inject altered controls, occluded keys and hidden stochastic switches
  - [ ] `AURA-07.3-T1-S2` Record whether system detects uncertainty before commitment
- [ ] **AURA-07.3-T2 — Quantify abstention costs**
  - [ ] `AURA-07.3-T2-S1` Report abstention frequency and post-abstention recovery outcomes
  - [ ] `AURA-07.3-T2-S2` Retain false positives and false negatives in the provenance ledger

**Section review gate:** Reconcile the above evidence with Section 07 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-08"></a>
### Section 08 — Curiosity

**Carried status:** Partial · **Priority:** P0 · **Related gates:** 06, 07, 12, 19, 24

#### AURA-08.1 — Intrinsic learning objectives

- [ ] **AURA-08.1-T1 — Distinguish useful information gain**
  - [ ] `AURA-08.1-T1-S1` Optimize validated learning progress rather than raw sensory novelty alone
  - [ ] `AURA-08.1-T1-S2` Ablate disagreement, novelty, noise penalty and information cost
- [ ] **AURA-08.1-T2 — Demonstrate non-oracle goal selection**
  - [ ] `AURA-08.1-T2-S1` Form investigative goals from observations without reading map metadata
  - [ ] `AURA-08.1-T2-S2` Compare success against random/fixed exploration with equal actions

#### AURA-08.2 — Localization and robust goal acquisition

- [ ] **AURA-08.2-T1 — Diagnose camera-motion failure**
  - [ ] `AURA-08.2-T1-S1` Replay the v117/v131 failures at 13x13/15x15 and 15–20% noise
  - [ ] `AURA-08.2-T1-S2` Partition failures into ambiguous calibration, movement mismatch and missing landmarks
- [ ] **AURA-08.2-T2 — Improve uncertainty-aware movement inference**
  - [ ] `AURA-08.2-T2-S1` Learn action effects from observations with confidence and rollback
  - [ ] `AURA-08.2-T2-S2` Check no evaluator-only position enters policy updates or action choice

#### AURA-08.3 — Generalization and ablations

- [ ] **AURA-08.3-T1 — Test truly unseen environment families**
  - [ ] `AURA-08.3-T1-S1` Freeze policy before sealed new sizes, rules, corruptions and movement transformations
  - [ ] `AURA-08.3-T1-S2` Include small, large, rare and adversarial conditions without cherry-picking
- [ ] **AURA-08.3-T2 — Compare navigation controllers fairly**
  - [ ] `AURA-08.3-T2-S1` Run persistent, composed, learned and random controllers on identical episodes
  - [ ] `AURA-08.3-T2-S2` Report paired discordances, clustered uncertainty and all negative outcomes

#### AURA-08.4 — Learning efficiency and external replication

- [ ] **AURA-08.4-T1 — Quantify cost of autonomous exploration**
  - [ ] `AURA-08.4-T1-S1` Track actions, wall-time, training updates, compute and retained skills per success
  - [ ] `AURA-08.4-T1-S2` Demonstrate improvement under matched total learning budget
- [ ] **AURA-08.4-T2 — Request blind external replication**
  - [ ] `AURA-08.4-T2-S1` Supply immutable source and evaluator interface to independent assessor
  - [ ] `AURA-08.4-T2-S2` Require signed complete raw episode receipts before Section 08 promotion

**Section review gate:** Reconcile the above evidence with Section 08 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-09"></a>
### Section 09 — Hypothesis and causal discovery

**Carried status:** Verified locally · **Priority:** P1 · **Related gates:** 10, 19, 24

#### AURA-09.1 — Hypothesis lifecycle

- [ ] **AURA-09.1-T1 — Replay established intervention studies**
  - [ ] `AURA-09.1-T1-S1` Verify stable, changing and masked camera cases with exact traces
  - [ ] `AURA-09.1-T1-S2` Audit rejected and abstained hypotheses as well as accepted ones
- [ ] **AURA-09.1-T2 — Expand causal hypothesis language**
  - [ ] `AURA-09.1-T2-S1` Add bounded typed variables for doors, switches and movement interventions
  - [ ] `AURA-09.1-T2-S2` Forbid post-hoc oracle access or dynamically invented unmeasured factors

#### AURA-09.2 — Experiment design

- [ ] **AURA-09.2-T1 — Compare interventions against observation**
  - [ ] `AURA-09.2-T1-S1` Randomize action assignments while holding initial conditions fixed
  - [ ] `AURA-09.2-T1-S2` Quantify confounding, budget and evidence gain per intervention
- [ ] **AURA-09.2-T2 — Add change-point hypotheses**
  - [ ] `AURA-09.2-T2-S1` Test rule switches and misleading observational correlations
  - [ ] `AURA-09.2-T2-S2` Require alternate-hypothesis falsification or explicit abstention

#### AURA-09.3 — Cross-domain causality

- [ ] **AURA-09.3-T1 — Transfer procedure to unfamiliar tasks**
  - [ ] `AURA-09.3-T1-S1` Reuse experiment planner in non-door mechanisms
  - [ ] `AURA-09.3-T1-S2` Benchmark causal identification against random experimentation
- [ ] **AURA-09.3-T2 — Verify counterfactual limits**
  - [ ] `AURA-09.3-T2-S1` Label simulated counterfactuals separately from observed outcomes
  - [ ] `AURA-09.3-T2-S2` Require actual follow-up trials before asserting a causal discovery

**Section review gate:** Reconcile the above evidence with Section 09 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-10"></a>
### Section 10 — Evidence & memory

**Carried status:** Verified locally · **Priority:** P2 · **Related gates:** 21, 29

#### AURA-10.1 — Ledger integrity

- [ ] **AURA-10.1-T1 — Reproduce anchored-evidence gate**
  - [ ] `AURA-10.1-T1-S1` Validate append-only hash chain, external pinned prefix and tamper rejection
  - [ ] `AURA-10.1-T1-S2` Check copied and truncated ledgers fail verified-prefix checks
- [ ] **AURA-10.1-T2 — Differentiate source categories**
  - [ ] `AURA-10.1-T2-S1` Record observed, imagined, external and derived evidence with provenance
  - [ ] `AURA-10.1-T2-S2` Reject cross-category writes that obscure origin

#### AURA-10.2 — Episodic/semantic/procedural retrieval

- [ ] **AURA-10.2-T1 — Validate three memory types**
  - [ ] `AURA-10.2-T1-S1` Store an episode, induced concept and learned skill with version links
  - [ ] `AURA-10.2-T1-S2` Reload after restart and check causal lineage and retrieval precision
- [ ] **AURA-10.2-T2 — Test contradiction management**
  - [ ] `AURA-10.2-T2-S1` Inject inconsistent evidence about action outcomes
  - [ ] `AURA-10.2-T2-S2` Ensure revisions retain old evidence and provide conflict-aware retrieval

#### AURA-10.3 — Capacity and audit

- [ ] **AURA-10.3-T1 — Bound storage growth**
  - [ ] `AURA-10.3-T1-S1` Measure retrieval latency and memory size under long runs
  - [ ] `AURA-10.3-T1-S2` Test archival/compaction without changing anchored evidence
- [ ] **AURA-10.3-T2 — Add export/replay tooling**
  - [ ] `AURA-10.3-T2-S1` Export evidence in stable machine-readable schema
  - [ ] `AURA-10.3-T2-S2` Verify independent reader can recompute digests and reconstruct decisions

**Section review gate:** Reconcile the above evidence with Section 10 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-11"></a>
### Section 11 — Concept discovery

**Carried status:** Verified locally · **Priority:** P2 · **Related gates:** 05, 10, 14

#### AURA-11.1 — Latent concept construction

- [ ] **AURA-11.1-T1 — Reproduce create/split/merge/retire gates**
  - [ ] `AURA-11.1-T1-S1` Rerun the five local shifted-observation experiments
  - [ ] `AURA-11.1-T1-S2` Preserve concept lineage, state snapshot and rejected edits
- [ ] **AURA-11.1-T2 — Eliminate shortcut labels**
  - [ ] `AURA-11.1-T2-S1` Generate candidate concepts from observed embeddings alone
  - [ ] `AURA-11.1-T2-S2` Measure dependence on hidden semantic labels or task metadata

#### AURA-11.2 — Concept quality

- [ ] **AURA-11.2-T1 — Compare against simple alternatives**
  - [ ] `AURA-11.2-T1-S1` Match compute for k-means-like prototypes and fixed random partitions
  - [ ] `AURA-11.2-T1-S2` Verify fresh held-out predictive gain with intervals
- [ ] **AURA-11.2-T2 — Audit false concept drift**
  - [ ] `AURA-11.2-T2-S1` Introduce nuisance perturbations and replay conflicts
  - [ ] `AURA-11.2-T2-S2` Distinguish useful refinement from unstable renaming

#### AURA-11.3 — Transfer and memory integration

- [ ] **AURA-11.3-T1 — Export concepts to skill and hypothesis users**
  - [ ] `AURA-11.3-T1-S1` Track where learned concepts change downstream decisions
  - [ ] `AURA-11.3-T1-S2` Test correct version lookup after concept retirement
- [ ] **AURA-11.3-T2 — Evaluate cross-task concepts**
  - [ ] `AURA-11.3-T2-S1` Freeze concepts before new task family evaluations
  - [ ] `AURA-11.3-T2-S2` Measure predictive and sample-efficiency effects versus no-concept controls

**Section review gate:** Reconcile the above evidence with Section 11 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-12"></a>
### Section 12 — Skills

**Carried status:** Verified locally · **Priority:** P1 · **Related gates:** 08, 10, 11, 19

#### AURA-12.1 — Skill discovery and grounding

- [ ] **AURA-12.1-T1 — Reproduce learned doorway skill**
  - [ ] `AURA-12.1-T1-S1` Replay shifted-layout and missing-key trials
  - [ ] `AURA-12.1-T1-S2` Verify success, abort behavior and withdrawal without oracle controls
- [ ] **AURA-12.1-T2 — Identify reusable primitives**
  - [ ] `AURA-12.1-T2-S1` Mine demonstrated options with initiation and termination tests
  - [ ] `AURA-12.1-T2-S2` Test that initiation criteria are sensory and not evaluator-derived

#### AURA-12.2 — Composition and safety

- [ ] **AURA-12.2-T1 — Validate option composition**
  - [ ] `AURA-12.2-T1-S1` Compose learned primitives for key-door-goal sequences
  - [ ] `AURA-12.2-T1-S2` Require fresh observed trials, not concatenation-only assertions
- [ ] **AURA-12.2-T2 — Detect failed skills**
  - [ ] `AURA-12.2-T2-S1` Inject blocked passages, changed movement and stale landmarks
  - [ ] `AURA-12.2-T2-S2` Ensure safe fallback, replan and negative skill revisions

#### AURA-12.3 — Transfer and efficiency

- [ ] **AURA-12.3-T1 — Evaluate unfamiliar contexts**
  - [ ] `AURA-12.3-T1-S1` Freeze skill library before held-out layouts and tasks
  - [ ] `AURA-12.3-T1-S2` Compare action cost and success versus flat policy
- [ ] **AURA-12.3-T2 — Track skill economics**
  - [ ] `AURA-12.3-T2-S1` Record acquisition cost, execution overhead and reuse savings
  - [ ] `AURA-12.3-T2-S2` Retire skills with repeated negative marginal value

**Section review gate:** Reconcile the above evidence with Section 12 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-13"></a>
### Section 13 — Curriculum

**Carried status:** Verified locally · **Priority:** P1 · **Related gates:** 08, 12, 15, 24

#### AURA-13.1 — Task generation

- [ ] **AURA-13.1-T1 — Build solvable learning progressions**
  - [ ] `AURA-13.1-T1-S1` Generate tasks with graded occlusion, length and rule complexity
  - [ ] `AURA-13.1-T1-S2` Verify solvability witnesses do not leak to learner
- [ ] **AURA-13.1-T2 — Audit challenge distribution**
  - [ ] `AURA-13.1-T2-S1` Cover easy, intermediate and adversarial tasks per seed
  - [ ] `AURA-13.1-T2-S2` Prevent curriculum from avoiding hard failures indefinitely

#### AURA-13.2 — Scheduling policies

- [ ] **AURA-13.2-T1 — Reproduce adaptive scheduler result**
  - [ ] `AURA-13.2-T1-S1` Rerun 20-seed transfer against fixed and random curricula
  - [ ] `AURA-13.2-T1-S2` Preserve documented negative neural-study finding separately
- [ ] **AURA-13.2-T2 — Compare under equal budgets**
  - [ ] `AURA-13.2-T2-S1` Normalize episodes, training updates and evaluator queries
  - [ ] `AURA-13.2-T2-S2` Report uncertainty and catastrophic transfer regressions

#### AURA-13.3 — Long-horizon learning

- [ ] **AURA-13.3-T1 — Link curriculum to mastery estimate**
  - [ ] `AURA-13.3-T1-S1` Revisit tasks after apparent progress to test retained competence
  - [ ] `AURA-13.3-T1-S2` Schedule corrective examples after detected forgetting
- [ ] **AURA-13.3-T2 — Test transfer scheduling**
  - [ ] `AURA-13.3-T2-S1` Train on procedural source family and test unfamiliar target family
  - [ ] `AURA-13.3-T2-S2` Establish gains beyond task memorization under frozen schedule

**Section review gate:** Reconcile the above evidence with Section 13 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-14"></a>
### Section 14 — Continual learning

**Carried status:** Verified locally · **Priority:** P1 · **Related gates:** 05, 10, 23

#### AURA-14.1 — Forgetting and rehearsal

- [ ] **AURA-14.1-T1 — Reproduce controlled rule-switch gate**
  - [ ] `AURA-14.1-T1-S1` Record old-task MSE before/after and new-task adaptation
  - [ ] `AURA-14.1-T1-S2` Compare replay, no-replay and equal-memory baselines
- [ ] **AURA-14.1-T2 — Manage replay quality**
  - [ ] `AURA-14.1-T2-S1` Balance recent vs representative examples and detect stale labels
  - [ ] `AURA-14.1-T2-S2` Preserve provenance when entries are evicted or sampled

#### AURA-14.2 — Distribution shifts

- [ ] **AURA-14.2-T1 — Stress multiple switches**
  - [ ] `AURA-14.2-T1-S1` Alternate action rules and observation channels over long sequences
  - [ ] `AURA-14.2-T1-S2` Quantify recovery time and accumulated forgetting
- [ ] **AURA-14.2-T2 — Validate transfer vs interference**
  - [ ] `AURA-14.2-T2-S1` Measure new-task and old-task metrics on identical saved banks
  - [ ] `AURA-14.2-T2-S2` Report when new learning improves one metric but damages another

#### AURA-14.3 — Resource efficiency

- [ ] **AURA-14.3-T1 — Compare memory budgets**
  - [ ] `AURA-14.3-T1-S1` Sweep replay buffer capacities with fixed compute
  - [ ] `AURA-14.3-T1-S2` Plot stability–memory tradeoff including wall-time and energy proxy
- [ ] **AURA-14.3-T2 — Check stable reinitialization**
  - [ ] `AURA-14.3-T2-S1` Restart learner during nonstationary training
  - [ ] `AURA-14.3-T2-S2` Verify optimizer, replay and curriculum state resume without data loss

**Section review gate:** Reconcile the above evidence with Section 14 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-15"></a>
### Section 15 — Meta-learning

**Carried status:** Verified locally · **Priority:** P1 · **Related gates:** 18, 23, 24

#### AURA-15.1 — Meta-controller actions

- [ ] **AURA-15.1-T1 — Reproduce scheduler improvement**
  - [ ] `AURA-15.1-T1-S1` Re-run 32-seed equal-update optimization comparison
  - [ ] `AURA-15.1-T1-S2` Audit source and target split leakage
- [ ] **AURA-15.1-T2 — Expand strategy choices**
  - [ ] `AURA-15.1-T2-S1` Expose bounded choices of exploration, replay, batch size and optimizer
  - [ ] `AURA-15.1-T2-S2` Enforce constraints and record which choice affected each update

#### AURA-15.2 — Meta-objective and credit assignment

- [ ] **AURA-15.2-T1 — Learn strategy utility from measured outcomes**
  - [ ] `AURA-15.2-T1-S1` Attribute held-out improvement to policy choices with conservative baselines
  - [ ] `AURA-15.2-T1-S2` Account for interactions and extra development-evaluation overhead
- [ ] **AURA-15.2-T2 — Add nonstationary adaptation**
  - [ ] `AURA-15.2-T2-S1` Switch preferred strategy after validated environment change
  - [ ] `AURA-15.2-T2-S2` Measure adaptation lag versus fixed policy

#### AURA-15.3 — Robust meta-transfer

- [ ] **AURA-15.3-T1 — Freeze meta-policy for new tasks**
  - [ ] `AURA-15.3-T1-S1` Train scheduler on source tasks and evaluate on different dynamics
  - [ ] `AURA-15.3-T1-S2` Compare budget-matched UCB, random and fixed schedules
- [ ] **AURA-15.3-T2 — Bound meta-learning overfit**
  - [ ] `AURA-15.3-T2-S1` Limit evaluation queries and preregister the split
  - [ ] `AURA-15.3-T2-S2` Reject strategies selected solely by repeated holdout peeking

**Section review gate:** Reconcile the above evidence with Section 15 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-16"></a>
### Section 16 — Recursive self-improvement

**Carried status:** Partial · **Priority:** P0 · **Related gates:** 17, 22, 25

#### AURA-16.1 — Bounded proposals

- [ ] **AURA-16.1-T1 — Enumerate typed mutation operators**
  - [ ] `AURA-16.1-T1-S1` Allow model, policy, curriculum and hyperparameter modifications through validated schemas
  - [ ] `AURA-16.1-T1-S2` Refuse arbitrary code paths, verifier files and threshold edits
- [ ] **AURA-16.1-T2 — Version candidate lineage**
  - [ ] `AURA-16.1-T2-S1` Save parent hash, diff, dependency lock and proposal rationale
  - [ ] `AURA-16.1-T2-S2` Verify failure, rollback and duplicate-detection paths

#### AURA-16.2 — Controlled evaluation

- [ ] **AURA-16.2-T1 — Evaluate one candidate against its parent**
  - [ ] `AURA-16.2-T1-S1` Run paired seeds with equal interactions and total compute
  - [ ] `AURA-16.2-T1-S2` Record confidence intervals, worst-case regressions and costs
- [ ] **AURA-16.2-T2 — Protect against selection bias**
  - [ ] `AURA-16.2-T2-S1` Segregate development, selection and untouched confirmation banks
  - [ ] `AURA-16.2-T2-S2` Apply repeated-look correction or report exploratory confidence only

#### AURA-16.3 — Closed-loop continuation

- [ ] **AURA-16.3-T1 — Integrate proposal loop with real learner**
  - [ ] `AURA-16.3-T1-S1` Trigger proposal from measured training deficiency and policy telemetry
  - [ ] `AURA-16.3-T1-S2` Execute and evaluate in isolation, then choose keep/reject explicitly
- [ ] **AURA-16.3-T2 — Demonstrate successive improvement**
  - [ ] `AURA-16.3-T2-S1` Produce two consecutive accepted generations under frozen criteria
  - [ ] `AURA-16.3-T2-S2` Confirm neither depends on score leakage or changed verifier behavior

#### AURA-16.4 — Safety and reversibility

- [ ] **AURA-16.4-T1 — Roll back failed descendants**
  - [ ] `AURA-16.4-T1-S1` Restore full model, optimizer, replay and lineage state after a rejection
  - [ ] `AURA-16.4-T1-S2` Verify replay consistency and no partial promotion artifact
- [ ] **AURA-16.4-T2 — Compare evolutionary alternatives**
  - [ ] `AURA-16.4-T2-S1` Benchmark random search, fixed schedule and informed proposal policies
  - [ ] `AURA-16.4-T2-S2` Retain costs of failed proposals and divergent search paths

**Section review gate:** Reconcile the above evidence with Section 16 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.
---

<a id="section-17"></a>
### Section 17 — Independent verifier

**Carried status:** Partial · **Priority:** P0 · **Related gates:** 02, 21, 25, 28

#### AURA-17.1 — Trust boundaries

- [ ] **AURA-17.1-T1 — Map all candidate influence channels**
  - [ ] `AURA-17.1-T1-S1` Enumerate filesystem, environment variables, imports, IPC, shared memory and network
  - [ ] `AURA-17.1-T1-S2` Define fail-closed trust model and prohibited evaluator writes
- [ ] **AURA-17.1-T2 — Verify isolated worker startup**
  - [ ] `AURA-17.1-T2-S1` Execute actual evaluator under separate process/container credentials
  - [ ] `AURA-17.1-T2-S2` Assert immutable bind mounts and denial of host-level secret access

#### AURA-17.2 — Sandbox and resource enforcement

- [ ] **AURA-17.2-T1 — Finish namespace/OS isolation integration**
  - [ ] `AURA-17.2-T1-S1` Link trusted PID/user/mount/seccomp probes to production worker
  - [ ] `AURA-17.2-T1-S2` Run kernel-specific probes and document unavailable protections
- [ ] **AURA-17.2-T2 — Exhaust adversarial mutation attempts**
  - [ ] `AURA-17.2-T2-S1` Inject symlink, path traversal, malformed JSON, inherited descriptor and syscall bypass inputs
  - [ ] `AURA-17.2-T2-S2` Prove no protected metadata or code was modified

#### AURA-17.3 — Independent evaluation protocol

- [ ] **AURA-17.3-T1 — Require external evaluator custody**
  - [ ] `AURA-17.3-T1-S1` Identify independent assessor with own sealed tasks and signing key
  - [ ] `AURA-17.3-T1-S2` Verify key fingerprint and affiliation through an out-of-band channel
- [ ] **AURA-17.3-T2 — Validate signed raw evidence**
  - [ ] `AURA-17.3-T2-S1` Check complete trial count, manifest hash, source commit and signature
  - [ ] `AURA-17.3-T2-S2` Independently replay selected outcomes and inspect conflicting records

#### AURA-17.4 — Promotion integrity

- [ ] **AURA-17.4-T1 — Verify immutable acceptance policy**
  - [ ] `AURA-17.4-T1-S1` Snapshot comparison thresholds and minimum sample sizes before candidate search
  - [ ] `AURA-17.4-T1-S2` Deny policy changes and repeated peeking without statistical accounting
- [ ] **AURA-17.4-T2 — Audit rejection paths**
  - [ ] `AURA-17.4-T2-S1` Test tiny samples, negative gains, nonfinite evidence and corrupt ledgers
  - [ ] `AURA-17.4-T2-S2` Ensure every rejected candidate leaves a durable non-promoted decision

**Section review gate:** Reconcile the above evidence with Section 17 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-18"></a>
### Section 18 — Value of computation

**Carried status:** Verified locally · **Priority:** P1 · **Related gates:** 15, 23, 24

#### AURA-18.1 — Benefit and cost measurement

- [ ] **AURA-18.1-T1 — Reproduce batch-allocation results**
  - [ ] `AURA-18.1-T1-S1` Re-run 32 controlled seeds and same-sample-budget comparisons
  - [ ] `AURA-18.1-T1-S2` Record prediction gain, compute cost and discarded candidates
- [ ] **AURA-18.1-T2 — Define resource cost model**
  - [ ] `AURA-18.1-T2-S1` Track wall-time, FLOP proxy, memory and simulator interactions
  - [ ] `AURA-18.1-T2-S2` Calibrate proxy against real CPU/GPU observations

#### AURA-18.2 — Decision policy

- [ ] **AURA-18.2-T1 — Compare compute allocations**
  - [ ] `AURA-18.2-T1-S1` Test small/frequent vs large/sparse update batches
  - [ ] `AURA-18.2-T1-S2` Use fixed random, greedy and best-known static baselines
- [ ] **AURA-18.2-T2 — Bound evaluation overhead**
  - [ ] `AURA-18.2-T2-S1` Account for value-estimation and scheduler queries
  - [ ] `AURA-18.2-T2-S2` Reject policies whose overhead exceeds measured learning benefit

#### AURA-18.3 — Generalization and efficiency

- [ ] **AURA-18.3-T1 — Transfer compute policy**
  - [ ] `AURA-18.3-T1-S1` Freeze allocation rule before unfamiliar tasks and model sizes
  - [ ] `AURA-18.3-T1-S2` Verify gains persist at matched total resource budgets
- [ ] **AURA-18.3-T2 — Analyze delayed outcomes**
  - [ ] `AURA-18.3-T2-S1` Track whether short-run gain harms later retention or transfer
  - [ ] `AURA-18.3-T2-S2` Report confidence and high-cost outlier failures

**Section review gate:** Reconcile the above evidence with Section 18 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-19"></a>
### Section 19 — Simulation environment

**Carried status:** Verified locally · **Priority:** P1 · **Related gates:** 03, 21, 24

#### AURA-19.1 — Environment correctness

- [ ] **AURA-19.1-T1 — Reproduce reachability witnesses**
  - [ ] `AURA-19.1-T1-S1` Replay all 90 disjoint-layout witnesses against environment dynamics
  - [ ] `AURA-19.1-T1-S2` Verify reward, termination, truncation and key/door invariants
- [ ] **AURA-19.1-T2 — Complete official API validation**
  - [ ] `AURA-19.1-T2-S1` Run Gymnasium checker, gym.make/EnvSpec and render RNG tests
  - [ ] `AURA-19.1-T2-S2` Confirm deterministic resets and correct observation/action spaces

#### AURA-19.2 — Procedural diversity

- [ ] **AURA-19.2-T1 — Generate novel solvable layouts**
  - [ ] `AURA-19.2-T1-S1` Vary size, wall geometry, occlusion, keys and door positions
  - [ ] `AURA-19.2-T1-S2` Independently verify generated levels are reachable before scoring learners
- [ ] **AURA-19.2-T2 — Stress action semantics**
  - [ ] `AURA-19.2-T2-S1` Randomize movement permutation, switchable rules and hidden hazards
  - [ ] `AURA-19.2-T2-S2` Ensure action effects require observation-based calibration

#### AURA-19.3 — Transfer and independence

- [ ] **AURA-19.3-T1 — Reproduce changed-action policy transfer**
  - [ ] `AURA-19.3-T1-S1` Verify 40/40 locally learned-calibration successes vs fixed-control failures
  - [ ] `AURA-19.3-T1-S2` Repeat on concealed layouts under independent assessor control
- [ ] **AURA-19.3-T2 — Prevent environment oracle leakage**
  - [ ] `AURA-19.3-T2-S1` Run policy with only RGB, discrete actions and permitted rewards
  - [ ] `AURA-19.3-T2-S2` Probe code and logs for direct access to map, hidden seeds and ground-truth state

**Section review gate:** Reconcile the above evidence with Section 19 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-20"></a>
### Section 20 — Repository structure

**Carried status:** Verified locally · **Priority:** P2 · **Related gates:** 29

#### AURA-20.1 — Package and filesystem layout

- [ ] **AURA-20.1-T1 — Confirm required directories**
  - [ ] `AURA-20.1-T1-S1` Audit aura, protected_verifier, configs, scripts, tests and docs tree
  - [ ] `AURA-20.1-T1-S2` Assert pip editable install and import paths in a fresh checkout
- [ ] **AURA-20.1-T2 — Define ownership boundaries**
  - [ ] `AURA-20.1-T2-S1` Mark verifier-owned and learner-owned paths
  - [ ] `AURA-20.1-T2-S2` Restrict untrusted code from modifying protected or externally held assets

#### AURA-20.2 — Reproducible experiments

- [ ] **AURA-20.2-T1 — Standardize run directory schema**
  - [ ] `AURA-20.2-T1-S1` Use versioned config, metrics, trace, checkpoint and hash manifest
  - [ ] `AURA-20.2-T1-S2` Validate all generated artifacts live outside source package and have run IDs
- [ ] **AURA-20.2-T2 — Minimize untracked dependencies**
  - [ ] `AURA-20.2-T2-S1` Enumerate required data files including historic Section 08 fixtures
  - [ ] `AURA-20.2-T2-S2` Add exact-hash reproducible regeneration for any missing public fixture

#### AURA-20.3 — Maintainable structure

- [ ] **AURA-20.3-T1 — Enforce module cohesion**
  - [ ] `AURA-20.3-T1-S1` Locate duplicated experimental navigation code and document interfaces
  - [ ] `AURA-20.3-T1-S2` Refactor only behind tests preserving historical reproducibility
- [ ] **AURA-20.3-T2 — Document package map**
  - [ ] `AURA-20.3-T2-S1` Maintain entrypoint and dependency diagram per major module
  - [ ] `AURA-20.3-T2-S2` Confirm new contributors can run bounded smoke without hidden setup

**Section review gate:** Reconcile the above evidence with Section 20 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-21"></a>
### Section 21 — Shared typed interfaces

**Carried status:** Verified locally · **Priority:** P2 · **Related gates:** 20

#### AURA-21.1 — Schema validation

- [ ] **AURA-21.1-T1 — Reproduce contract tests**
  - [ ] `AURA-21.1-T1-S1` Instantiate Experience, BeliefState, Hypothesis, Skill, Modification and EvaluationReport
  - [ ] `AURA-21.1-T1-S2` Reject invalid shapes, provenance, dtypes and missing fields
- [ ] **AURA-21.1-T2 — Freeze wire-compatible versions**
  - [ ] `AURA-21.1-T2-S1` Version persistent data structures and ledger records
  - [ ] `AURA-21.1-T2-S2` Test backward migration and explicit rejection of unsupported schema

#### AURA-21.2 — Runtime interoperability

- [ ] **AURA-21.2-T1 — Test producer-consumer boundaries**
  - [ ] `AURA-21.2-T1-S1` Carry observations and beliefs through model, memory and actor
  - [ ] `AURA-21.2-T1-S2` Verify no implicit untyped dict conversions bypass guards
- [ ] **AURA-21.2-T2 — Preserve observed vs imagined origin**
  - [ ] `AURA-21.2-T2-S1` Label every policy/planning data transfer by evidence source
  - [ ] `AURA-21.2-T2-S2` Reject mutation or coercion of provenance at handoff

#### AURA-21.3 — Evolution and diagnostics

- [ ] **AURA-21.3-T1 — Add contract fixtures**
  - [ ] `AURA-21.3-T1-S1` Save minimal valid/invalid records in test data
  - [ ] `AURA-21.3-T1-S2` Run roundtrip JSON/SQLite/checkpoint serialization tests
- [ ] **AURA-21.3-T2 — Provide structured errors**
  - [ ] `AURA-21.3-T2-S1` Return explicit schema failures with source and field names
  - [ ] `AURA-21.3-T2-S2` Ensure external APIs do not expose protected secrets in errors

**Section review gate:** Reconcile the above evidence with Section 21 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-22"></a>
### Section 22 — Core integrated loop

**Carried status:** Partial · **Priority:** P0 · **Related gates:** 04, 06, 08, 10, 12, 13, 15, 16

#### AURA-22.1 — Online interaction loop

- [ ] **AURA-22.1-T1 — Trace sensory-action-update sequence**
  - [ ] `AURA-22.1-T1-S1` Run reset/observe/predict/act/step/log under a deterministic seed
  - [ ] `AURA-22.1-T1-S2` Verify on-policy decisions never read evaluator-only state
- [ ] **AURA-22.1-T2 — Integrate uncertainty-aware policy**
  - [ ] `AURA-22.1-T2-S1` Gate risky actions and invoke movement recalibration when needed
  - [ ] `AURA-22.1-T2-S2` Compare paired goal success and extra action costs

#### AURA-22.2 — Online learning and memory

- [ ] **AURA-22.2-T1 — Connect replay to model and skill updates**
  - [ ] `AURA-22.2-T1-S1` Log exactly which observations train encoder, world model and options
  - [ ] `AURA-22.2-T1-S2` Verify no invented transitions are promoted to evidence memory
- [ ] **AURA-22.2-T2 — Close curriculum feedback loop**
  - [ ] `AURA-22.2-T2-S1` Update task selection from measured learning outcomes
  - [ ] `AURA-22.2-T2-S2` Account for held-out feedback use as separate learning budget

#### AURA-22.3 — Recursive improvement orchestration

- [ ] **AURA-22.3-T1 — Invoke bounded proposal pipeline**
  - [ ] `AURA-22.3-T1-S1` Trigger candidate generation from a measurable deficiency
  - [ ] `AURA-22.3-T1-S2` Run protected verifier in a genuinely separate process
- [ ] **AURA-22.3-T2 — Implement commit/rollback state transitions**
  - [ ] `AURA-22.3-T2-S1` Promote only after independently valid report
  - [ ] `AURA-22.3-T2-S2` Restore all learner state on failure and retain failure lineage

#### AURA-22.4 — Recovery and long-run endurance

- [ ] **AURA-22.4-T1 — Run interrupted multi-hour smoke**
  - [ ] `AURA-22.4-T1-S1` Inject crashes, restarts, corrupt checkpoints and timeouts
  - [ ] `AURA-22.4-T1-S2` Verify resumed run matches deterministic reference on supported paths
- [ ] **AURA-22.4-T2 — Bound memory and compute**
  - [ ] `AURA-22.4-T2-S1` Enforce runtime quotas without silently dropping evidence
  - [ ] `AURA-22.4-T2-S2` Record clean shutdown and incomplete run status

**Section review gate:** Reconcile the above evidence with Section 22 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-23"></a>
### Section 23 — Training objectives

**Carried status:** Partial · **Priority:** P1 · **Related gates:** 06, 07, 15, 18

#### AURA-23.1 — Predictive/representation loss

- [ ] **AURA-23.1-T1 — Audit every term**
  - [ ] `AURA-23.1-T1-S1` Verify reconstruction, latent dynamics, image loss and anti-collapse weights
  - [ ] `AURA-23.1-T1-S2` Test gradients are finite and correct parameters receive updates
- [ ] **AURA-23.1-T2 — Validate horizon losses**
  - [ ] `AURA-23.1-T2-S1` Train multi-step residual heads with aligned sequences
  - [ ] `AURA-23.1-T2-S2` Compare against equal-update absolute and no-hierarchy controls

#### AURA-23.2 — Decision and control objectives

- [ ] **AURA-23.2-T1 — Reproduce actor TD target**
  - [ ] `AURA-23.2-T1-S1` Test reward, terminal mask, truncation bootstrap and gamma
  - [ ] `AURA-23.2-T1-S2` Check Q update improves policy on a held-out simple task
- [ ] **AURA-23.2-T2 — Optimize intrinsic objectives without reward hacking**
  - [ ] `AURA-23.2-T2-S1` Penalize TV noise and raw novelty exploit
  - [ ] `AURA-23.2-T2-S2` Compare genuine goal learning against sensory distractor fascination

#### AURA-23.3 — Resource/stability objectives

- [ ] **AURA-23.3-T1 — Evaluate cost-sensitive meta reward**
  - [ ] `AURA-23.3-T1-S1` Quantify learning gain per additional action/update/second
  - [ ] `AURA-23.3-T1-S2` Match budgets and avoid double counting evaluator queries
- [ ] **AURA-23.3-T2 — Guard optimization failures**
  - [ ] `AURA-23.3-T2-S1` Inject gradient explosion, NaN, stale replay and zero-variance features
  - [ ] `AURA-23.3-T2-S2` Enforce clipping, rejection and diagnostic preservation

**Section review gate:** Reconcile the above evidence with Section 23 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-24"></a>
### Section 24 — Benchmarks

**Carried status:** Partial · **Priority:** P0 · **Related gates:** 03, 06, 08, 19, 25

#### AURA-24.1 — Public regression baselines

- [ ] **AURA-24.1-T1 — Maintain fixed public matrix**
  - [ ] `AURA-24.1-T1-S1` Run random, fixed, model-based, persistent and composed baselines
  - [ ] `AURA-24.1-T1-S2` Keep per-trial action traces, costs, seeds, configurations and hashes
- [ ] **AURA-24.1-T2 — Preserve known counterexamples**
  - [ ] `AURA-24.1-T2-S1` Retain v117 and v131 composition reversals and severe-noise failures
  - [ ] `AURA-24.1-T2-S2` Add tests that fail when adverse trials are omitted or duplicated

#### AURA-24.2 — Held-out generalization

- [ ] **AURA-24.2-T1 — Preregister independent split**
  - [ ] `AURA-24.2-T1-S1` Freeze model and development tuning before new environments are drawn
  - [ ] `AURA-24.2-T1-S2` Hide evaluation seeds, task families and altered rules until execution
- [ ] **AURA-24.2-T2 — Evaluate transfer by regime**
  - [ ] `AURA-24.2-T2-S1` Stratify layouts, corruption, horizon, movement changes and task family
  - [ ] `AURA-24.2-T2-S2` Report rare failures and unsolved conditions, not only average success

#### AURA-24.3 — Statistical and compute validity

- [ ] **AURA-24.3-T1 — Use paired grouped inference**
  - [ ] `AURA-24.3-T1-S1` Bootstrap at independent environment/seed cluster level
  - [ ] `AURA-24.3-T1-S2` Report effect size, uncertainty, negative controls and multiple-look policy
- [ ] **AURA-24.3-T2 — Match training/testing budgets**
  - [ ] `AURA-24.3-T2-S1` Account for all model updates, additional diagnostics and probe actions
  - [ ] `AURA-24.3-T2-S2` Include per-controller wall-time and full search cost

#### AURA-24.4 — External benchmark certification

- [ ] **AURA-24.4-T1 — Obtain independent evaluator agreement**
  - [ ] `AURA-24.4-T1-S1` Supply immutable source, task interface and non-promoting receipt template
  - [ ] `AURA-24.4-T1-S2` Require evaluator-controlled hidden test generator and execution host
- [ ] **AURA-24.4-T2 — Review signed scorecards**
  - [ ] `AURA-24.4-T2-S1` Verify signed raw outcomes and reproducibility on third-party host
  - [ ] `AURA-24.4-T2-S2` Decide criterion by criterion and retain disputes without automatic promotion

**Section review gate:** Reconcile the above evidence with Section 24 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-25"></a>
### Section 25 — Recursive validation

**Carried status:** Open · **Priority:** P0 · **Related gates:** 16, 17, 24

#### AURA-25.1 — Two-generation candidate chain

- [ ] **AURA-25.1-T1 — Predeclare two-step improvement targets**
  - [ ] `AURA-25.1-T1-S1` Define performance, cost, safety and retention metrics for G0, G1 and G2
  - [ ] `AURA-25.1-T1-S2` Freeze evaluator policies and minimum gains before search
- [ ] **AURA-25.1-T2 — Execute full candidate lineage**
  - [ ] `AURA-25.1-T2-S1` Run G0→G1→G2 with real proposals and separate held-out evaluation
  - [ ] `AURA-25.1-T2-S2` Record every rejected sibling and rollback, not only winning descendant

#### AURA-25.2 — Statistical validation

- [ ] **AURA-25.2-T1 — Test paired gains independently**
  - [ ] `AURA-25.2-T1-S1` Use disjoint development and blind confirmation environments per generation
  - [ ] `AURA-25.2-T1-S2` Evaluate worst-seed regression and family-wise repeated-look risk
- [ ] **AURA-25.2-T2 — Check cumulative retained improvement**
  - [ ] `AURA-25.2-T2-S1` Rerun G0/G1/G2 on the same independent held-out families
  - [ ] `AURA-25.2-T2-S2` Reject chains where second gain erases earlier skills or inflates cost

#### AURA-25.3 — External adjudication

- [ ] **AURA-25.3-T1 — Appoint independent assessor**
  - [ ] `AURA-25.3-T1-S1` Obtain conflict-of-interest statement and assessor-owned task seeds
  - [ ] `AURA-25.3-T1-S2` Freeze software and verifier hashes prior to any evaluator access
- [ ] **AURA-25.3-T2 — Receive and audit full receipts**
  - [ ] `AURA-25.3-T2-S1` Validate signatures, hash manifests and all trial traces
  - [ ] `AURA-25.3-T2-S2` Require independent reruns and discrepancy resolution before acceptance

#### AURA-25.4 — Publication and gating

- [ ] **AURA-25.4-T1 — Produce reproducible chain dossier**
  - [ ] `AURA-25.4-T1-S1` Export full lineage, source, seeds escrow, clocks, budgets and negative trials
  - [ ] `AURA-25.4-T1-S2` Ensure a new auditor can reproduce evidence without private developer state
- [ ] **AURA-25.4-T2 — Keep gate Open until all conditions pass**
  - [ ] `AURA-25.4-T2-S1` Store a machine-readable pass/fail per required subcriterion
  - [ ] `AURA-25.4-T2-S2` Prohibit broad recursive-learning claims based only on one accepted child

**Section review gate:** Reconcile the above evidence with Section 25 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-26"></a>
### Section 26 — Execution phases

**Carried status:** Partial · **Priority:** P1 · **Related gates:** 01, 22, 24, 25, 30

#### AURA-26.1 — Formal phase map

- [ ] **AURA-26.1-T1 — Enumerate phases 1–7**
  - [ ] `AURA-26.1-T1-S1` Map each phase to current modules, prerequisites and missing milestones
  - [ ] `AURA-26.1-T1-S2` Distinguish functional smoke from research acceptance
- [ ] **AURA-26.1-T2 — Define phase exit criteria**
  - [ ] `AURA-26.1-T2-S1` Require unit, integration, transfer and audit deliverables per stage
  - [ ] `AURA-26.1-T2-S2` Prohibit advance based solely on source-file presence

#### AURA-26.2 — Progressive implementation

- [ ] **AURA-26.2-T1 — Execute prerequisites first**
  - [ ] `AURA-26.2-T1-S1` Prioritize runtime, verifier, world-model and exploration blockers
  - [ ] `AURA-26.2-T1-S2` Switch to other reachable gates if external review is delayed
- [ ] **AURA-26.2-T2 — Preserve controlled checkpoints**
  - [ ] `AURA-26.2-T2-S1` Tag source and bundle per validated phase or experiment
  - [ ] `AURA-26.2-T2-S2` Attach negative results and prior acceptance register to each version

#### AURA-26.3 — Final phase integration

- [ ] **AURA-26.3-T1 — Re-run cross-phase suite**
  - [ ] `AURA-26.3-T1-S1` Execute all phase-specific commands from an isolated checkout
  - [ ] `AURA-26.3-T1-S2` Validate no later phase weakens earlier protected constraints
- [ ] **AURA-26.3-T2 — Prepare final readiness review**
  - [ ] `AURA-26.3-T2-S1` Assemble phase-by-phase evidence and unresolved limitations
  - [ ] `AURA-26.3-T2-S2` Hold Phase 7 open until independent final acceptance succeeds

**Section review gate:** Reconcile the above evidence with Section 26 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-27"></a>
### Section 27 — Commands and artifacts

**Carried status:** Partial · **Priority:** P1 · **Related gates:** 20, 28, 29

#### AURA-27.1 — Supported commands

- [ ] **AURA-27.1-T1 — Verify training and evaluation commands**
  - [ ] `AURA-27.1-T1-S1` Run train, evaluate, recursive, minimum-architecture and held-out CLI entries
  - [ ] `AURA-27.1-T1-S2` Capture exact invocation, environment and process exit in every case
- [ ] **AURA-27.1-T2 — Test options and failures**
  - [ ] `AURA-27.1-T2-S1` Validate unknown task names, missing files, malformed configs and timeout handling
  - [ ] `AURA-27.1-T2-S2` Confirm nonzero exit with actionable diagnostics and no state promotion

#### AURA-27.2 — Artifact completeness

- [ ] **AURA-27.2-T1 — Standardize experiment output**
  - [ ] `AURA-27.2-T1-S1` Save run manifest, config, source SHA, checkpoints and raw per-trial records
  - [ ] `AURA-27.2-T1-S2` Create SHA-256 index and verify replay of representative trials
- [ ] **AURA-27.2-T2 — Capture failed executions durably**
  - [ ] `AURA-27.2-T2-S1` Persist partial logs and timed-out process state atomically
  - [ ] `AURA-27.2-T2-S2` Ensure CI uploads results on both pass and fail, with retention policy

#### AURA-27.3 — Public reproducibility

- [ ] **AURA-27.3-T1 — Test archive portability**
  - [ ] `AURA-27.3-T1-S1` Extract versioned source without installed local development tree
  - [ ] `AURA-27.3-T1-S2` Run documented commands from clean working directory
- [ ] **AURA-27.3-T2 — Publish reproduction index**
  - [ ] `AURA-27.3-T2-S1` Map every important acceptance claim to commands and artifact paths
  - [ ] `AURA-27.3-T2-S2` Label fallback-only, hosted official and externally independent evidence distinctly

**Section review gate:** Reconcile the above evidence with Section 27 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-28"></a>
### Section 28 — Quality and tests

**Carried status:** Partial · **Priority:** P0 · **Related gates:** 03, 17, 22, 24

#### AURA-28.1 — Reliable complete suite

- [ ] **AURA-28.1-T1 — Run durable hosted full suite**
  - [ ] `AURA-28.1-T1-S1` Execute all collected tests under Python 3.11/3.12
  - [ ] `AURA-28.1-T1-S2` Capture explicit exit code, JUnit XML, raw log and source commit for each matrix cell
- [ ] **AURA-28.1-T2 — Reconstruct test fixtures safely**
  - [ ] `AURA-28.1-T2-S1` Restore historical Section 08 data from deterministic generators
  - [ ] `AURA-28.1-T2-S2` Match original content hashes and fail rather than silently alter tests

#### AURA-28.2 — Official and optional platform gates

- [ ] **AURA-28.2-T1 — Validate Gymnasium and rendering**
  - [ ] `AURA-28.2-T1-S1` Run official checker and registered env tests with actual installed backend
  - [ ] `AURA-28.2-T1-S2` Test render RNG purity, observation spaces, reset/step and termination semantics
- [ ] **AURA-28.2-T2 — Exercise privileged optional probes**
  - [ ] `AURA-28.2-T2-S1` Run namespace/seccomp tests on a compatible authorized Linux worker
  - [ ] `AURA-28.2-T2-S2` Distinguish skipped unsupported capabilities from verified protections

#### AURA-28.3 — Adversarial and restart testing

- [ ] **AURA-28.3-T1 — Protect verifier invariants**
  - [ ] `AURA-28.3-T1-S1` Fuzz invalid JSON, descriptor leaks, symlinks and policy-change attempts
  - [ ] `AURA-28.3-T1-S2` Preserve rejected inputs and verify protected files are unchanged
- [ ] **AURA-28.3-T2 — Validate runtime recovery**
  - [ ] `AURA-28.3-T2-S1` Kill and restart mid-evaluation, mid-ledger write and mid-training
  - [ ] `AURA-28.3-T2-S2` Compare persisted state and ensure no phantom successes

#### AURA-28.4 — Continuous regression management

- [ ] **AURA-28.4-T1 — Add stress regression inventory**
  - [ ] `AURA-28.4-T1-S1` Pin v117/v131 counterexamples and Section 08 hard seeds
  - [ ] `AURA-28.4-T1-S2` Require CI audit of failures even when aggregate performance improves
- [ ] **AURA-28.4-T2 — Track coverage and flaky behavior**
  - [ ] `AURA-28.4-T2-S1` Run independent repetitions with controlled time limits
  - [ ] `AURA-28.4-T2-S2` Separate actual flaky tests from environment interruptions and archive every exit

**Section review gate:** Reconcile the above evidence with Section 28 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-29"></a>
### Section 29 — Repository workflow

**Carried status:** Partial · **Priority:** P1 · **Related gates:** 20, 27, 28

#### AURA-29.1 — Branch and pull-request discipline

- [ ] **AURA-29.1-T1 — Preserve protected default branch**
  - [ ] `AURA-29.1-T1-S1` Develop on separate integration branches with auditable PR diffs
  - [ ] `AURA-29.1-T1-S2` Require explicit review before merge and reject force-push of history
- [ ] **AURA-29.1-T2 — Document upstream/downstream ancestry**
  - [ ] `AURA-29.1-T2-S1` Verify required f582278, 2c8f161, 8fe6ce2 and f0ab324 reachable in source history
  - [ ] `AURA-29.1-T2-S2` Preserve concurrent descendants when selecting newest integrated checkpoint

#### AURA-29.2 — CI and release hygiene

- [ ] **AURA-29.2-T1 — Enforce required quality checks**
  - [ ] `AURA-29.2-T1-S1` Attach Python matrix, official backend, negative fixtures and trusted verifier checks
  - [ ] `AURA-29.2-T1-S2` Fail the release gate when an expected run is skipped or missing
- [ ] **AURA-29.2-T2 — Package versioned checkpoint**
  - [ ] `AURA-29.2-T2-S1` Publish source ZIP, complete Git bundle and manifest
  - [ ] `AURA-29.2-T2-S2` Verify archive integrity, reproducible commands and citation to original commit

#### AURA-29.3 — Issue and review management

- [ ] **AURA-29.3-T1 — Create acceptance-linked backlog**
  - [ ] `AURA-29.3-T1-S1` Cross-link work-breakdown IDs, PRs, issues and test evidence
  - [ ] `AURA-29.3-T1-S2` Keep blocked external work distinct from code-ready subtasks
- [ ] **AURA-29.3-T2 — Apply conservative review decisions**
  - [ ] `AURA-29.3-T2-S1` Compare implementation results to original acceptance register
  - [ ] `AURA-29.3-T2-S2` Retain draft status if independent evidence or security integration is missing

**Section review gate:** Reconcile the above evidence with Section 29 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-30"></a>
### Section 30 — Final acceptance

**Carried status:** Open · **Priority:** P0 · **Related gates:** 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31

#### AURA-30.1 — Per-section review

- [ ] **AURA-30.1-T1 — Audit 31 original criteria**
  - [ ] `AURA-30.1-T1-S1` Match each section to its exact original requirement, test and signed evidence
  - [ ] `AURA-30.1-T1-S2` Record explicit pass, partial, open or contested outcome with rationale
- [ ] **AURA-30.1-T2 — Reconcile historical reports**
  - [ ] `AURA-30.1-T2-S1` Validate local gates, source ancestry and externally pinned evidence
  - [ ] `AURA-30.1-T2-S2` Keep later failed replications and contradictory results in the final dossier

#### AURA-30.2 — End-to-end independent assessment

- [ ] **AURA-30.2-T1 — Rebuild and rerun full system externally**
  - [ ] `AURA-30.2-T1-S1` Have a separate assessor execute a frozen source and hidden task battery
  - [ ] `AURA-30.2-T1-S2` Audit all raw outcomes, budgets, failures, evaluator privileges and signatures
- [ ] **AURA-30.2-T2 — Verify recursive-generalization claim**
  - [ ] `AURA-30.2-T2-S1` Demand two independently accepted improvement generations and broad transfer
  - [ ] `AURA-30.2-T2-S2` Reject declarations based on simulation-only or controlled development tasks

#### AURA-30.3 — Completion and pause

- [ ] **AURA-30.3-T1 — Publish final audit package**
  - [ ] `AURA-30.3-T1-S1` Release source, full history, manifests, accepted/rejected candidates and reproducibility guide
  - [ ] `AURA-30.3-T1-S2` Compute cryptographic hashes and sign independent evaluation references
- [ ] **AURA-30.3-T2 — Stop completion automation on proof**
  - [ ] `AURA-30.3-T2-S1` Check all 31 section gates explicitly and require an externally reviewed final pass
  - [ ] `AURA-30.3-T2-S2` Archive final report and pause the hourly completion task only after full proof

**Section review gate:** Reconcile the above evidence with Section 30 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

<a id="section-31"></a>
### Section 31 — Engineering principle

**Carried status:** Ongoing · **Priority:** P1 · **Related gates:** 02, 17, 25, 29

#### AURA-31.1 — Evidence over claims

- [ ] **AURA-31.1-T1 — Maintain claim-to-evidence policy**
  - [ ] `AURA-31.1-T1-S1` Require each improvement statement to cite reproducible test records
  - [ ] `AURA-31.1-T1-S2` Reject AGI, autonomous-general-learning or security claims beyond demonstrated scope
- [ ] **AURA-31.1-T2 — Preserve negative learning outcomes**
  - [ ] `AURA-31.1-T2-S1` Archive failed candidates, unexpected regressions and interrupted runs
  - [ ] `AURA-31.1-T2-S2` Make trend summaries include failures and evaluated sample counts

#### AURA-31.2 — Conservative change selection

- [ ] **AURA-31.2-T1 — Prefer smallest measurable next change**
  - [ ] `AURA-31.2-T1-S1` Compare proposed complexity, risk and evidence value against baseline
  - [ ] `AURA-31.2-T1-S2` Select prerequisites before speculative system rewrites
- [ ] **AURA-31.2-T2 — Require safe reversible experiments**
  - [ ] `AURA-31.2-T2-S1` Gate changes behind typed configs, tests and separate branches
  - [ ] `AURA-31.2-T2-S2` Roll back regressions without modifying trusted verifier state

#### AURA-31.3 — Long-term reproducibility and stewardship

- [ ] **AURA-31.3-T1 — Continue periodic execution with audit trail**
  - [ ] `AURA-31.3-T1-S1` Select highest-impact reachable task from the WBS each run
  - [ ] `AURA-31.3-T1-S2` Record current commit, command, outcome, blockers and next dependency
- [ ] **AURA-31.3-T2 — Terminate only on independently supported closure**
  - [ ] `AURA-31.3-T2-S1` Maintain explicit 31/31 audit and protected stop condition
  - [ ] `AURA-31.3-T2-S2` Pause automation after final verified evidence, not after controlled or partial success

**Section review gate:** Reconcile the above evidence with Section 31 in `docs/section_acceptance.md`, capture discrepancies and negative results, and obtain the required review before any status change.

---

## Recommended tracking fields

In the companion CSV/JSON, stable IDs are usable as issue titles, progress markers, or automation handoff keys. Use `execution_status` values `unassessed`, `active`, `blocked`, `evidence-ready`, `independently-verified` and `rejected`; **none** is inferred automatically from the acceptance register. Link every subtask to its Git commit, reproducible command, evidence manifest and reviewer.

## Deferred external dependencies

The following cannot be self-certified by author-controlled code: a genuinely separate evaluator with custody of hidden scenarios, an independently authenticated signing key, hardware/kernel-dependent production isolation, and a third-party audit of successive accepted generations. Issues or automated signatures are **not** substitutes for an external assessor.
