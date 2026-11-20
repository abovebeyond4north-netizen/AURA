# Verification report — 2026-11-16

**Local suite:** 39 tests passed on Python 3.13.5; smoke learning executed with CPU PyTorch.

## Matched interaction-budget short run (5 seeds × 80 interactions)

| Condition | Held-out prediction MSE, mean | Task success |
|---|---:|---:|
| random | 0.157109 | 0 observed wins |
| fixed | 0.051922 | 0 observed wins |
| curiosity | 0.044354 | 0 observed wins |
| adaptive | 0.045981 | 0 observed wins |
| omega | 0.045981 | 0 observed wins |

These are small, descriptive runs, not proof of statistically significant relative performance. The omega and adaptive conditions used the same learning policy in this study; recursive configuration search was evaluated separately. Task schedules differ for adaptive curriculum conditions, so the numbers are not a clean isolated causal ablation.

## Noisy-TV benchmark (5 seeds, 350 choices each)

| Action policy | Mean fraction choosing noise in last 100 decisions |
|---|---:|
| random | 0.470 |
| prediction_error | 0.986 |
| learning_progress | 0.030 |

The benchmark is a controlled two-action Bernoulli simulator. The noise-aware policy penalizes held-out unpredictable outcomes; superiority here does not yet show performance across uncontrolled visual environments.

## Sequential-task retention diagnostic (3 seeds)

Both the replay and no-replay variants improved their old-task next-image prediction after learning the altered rule. This run does **not** establish that replay prevented forgetting.

## Bounded two-stage self-improvement trial

The current lineage contains a rejected earlier candidate and a subsequent accepted learning-rate change (`experiment-2`). Its successor (`experiment-3`) was rejected. The latest run showed:

- First-generation paired bootstrap 95% interval approximately **[0.000384, 0.002894]** image-MSE gain.
- Second-generation paired bootstrap 95% interval approximately **[-0.002945, -0.000346]** image-MSE gain.

Because protected evaluation seeds were exposed to repeated testing, these intervals do not support confirmatory significance claims. **Two successive improvements were not demonstrated.**

## Resource and provenance checks

Smoke run: 80 interactions; 139 optimization/concept steps; 0.835 s measured wall time; 323.4 MiB process RSS; ledger verified `True`.

### Unmet acceptance criteria

No achieved gridworld goal, no independently measured skill composition or transfer, no causal mechanism identification under partial perception, no demonstrated two-generation improvement, no OS-enforced verifier sandbox, and no large-scale statistically corrected evaluation.

## Incremental engineering validation — continuation, 2026-11-16

A fresh continuation recovered local commit `10c4d90` and reran the original 39 tests. The expanded local suite subsequently passed **57 tests** on Python 3.13.5. The added tests cover controlled conditional-option learning, randomized causal interventions, action-sequence temporal prediction, integration/checkpointing, horizon-specific held-out evaluation, and evaluator-only proof of task reachability for all five main environment families.

### Controlled unseen composition (five mechanism seeds)

After 80 single-stage reward-driven exploration trials per primitive (160 per seed), evaluation on a new two-stage task used **60 test episodes per seed** (300 total):

| Policy | Observed successes | Success rate | Wilson 95% interval |
|---|---:|---:|---|
| Image-prototype-conditioned learned options | 300 / 300 | 100.0% | [98.74%, 100.0%] |
| Random primitive actions | 10 / 300 | 3.33% | [1.82%, 6.03%] |

These intervals describe episode-level binomial sampling conditional on the five deterministic mechanism seeds; they do not correct for correlations within mechanism or establish transfer to unfamiliar action mappings. Each primitive is recognized through an unlabeled RGB prototype, and its action is chosen from reward data, not a hardcoded simulator solution. **This demonstrates tightly scoped composition transfer, not gridworld task mastery.** Full raw per-seed outputs are saved as `sample_outputs/controlled_mechanisms.json`.

### Controlled causal study

Using 400 trials per study, passive confounded actions produced a strong misleading association. With randomized interventions, the confounded case was classified **NEGLIGIBLE_EFFECT_SUPPORTED**, with observed effect +0.085 and normal-approximation interval [-0.013, 0.183]. An effective mechanism was classified **POSITIVE_EFFECT_SUPPORTED**, with effect +0.890 and interval [0.845, 0.935]. These are the measured values from the saved benchmark using seed 7; unit tests use separate seeds. The design requires stable mechanism, valid randomization and reliable sensor; these prerequisites would not automatically hold in unknown real environments.

### Multi-horizon temporal heads

The normal 80-step CPU smoke training completed **16 additional direct temporal-head updates** and produced `temporal.pt`. A separate held-out development evaluation returned image MSE for 1/2/4-step horizons across all five task families, recorded in `sample_outputs/development_temporal_evaluation.json`. The temporal predictions remain worse than the trained one-step predictor under the tiny smoke budget, so the additional hierarchy is an **implemented experimental component, not a validated improvement**.

### Reachability proof

An evaluator-only breadth-first search found and replayed valid solution action sequences for all five supported gridworld task families from test seed 3. The training agent cannot access this privileged planning function or diagnostic state. This establishes reachability of the tested layouts, not all possible procedural seeds.

The master 31-section acceptance register is in `docs/section_acceptance.md`. Major remaining gaps are still open.

### Resource exhaustion and computation selection

Two additional tests demonstrate that the main training loop does not attempt optimization after an exhausted update quota, and that an online value-of-computation heuristic can select a lower-cost operator or stop when measured net gains are negative. This is an isolated functional test, **not** a demonstrated end-to-end improvement in learning efficiency.


## Incremental engineering validation — multiscale prediction and uncertainty

After additional action-residual, learned-latent and calibrated-shift integration, **69 automated tests passed** locally using `python -m pytest -q`. This is a development/test validation statement, not independent research certification. The new tests exercise checkpoint v1-to-v3 compatibility, observed-versus-unobserved conditional outcomes, frozen shift thresholds, and planner horizon contraction on detected shifts.

### Controlled multiscale prediction (five seeds)

All models used the same frozen, randomly initialized CNN encoder, separate image and latent prediction heads, 300 training updates, three observed training trajectories and three independently seeded held-out trajectories per seed. Initial residual predictions exactly copy the current image. All numbers are mean held-out normalized image MSE (lower is better).

| Horizon | Persistence baseline | Trained absolute-image head | Trained residual head |
|---|---:|---:|---:|
| 1 | 0.002050 | 0.020483 | 0.000107 |
| 2 | 0.005304 | 0.019216 | 0.000774 |
| 4 | 0.016488 | 0.019813 | 0.003293 |

The trained residual latent-effect heads also improved against their own initial latent persistence reference at each horizon on every seed. Full records appear in `sample_outputs/multiscale_report.json`. This process demonstrates controlled predictive multi-horizon learning, but its additive-channel rule is intentionally simple and cannot establish abstract transfer.

### Gridworld development evaluation

An 80-interaction CPU smoke run added 16 temporal training updates. The 1-step residual temporal image error on key tasks was 0.0484, compared with main one-step model 0.0424, and the noisy visual task remained harder. Results are saved as `sample_outputs/development_multiscale_evaluation.json`. **No gridworld temporal-head advantage is claimed.** The residual/latent checkpoint now uses schema v3; schema v1 absolute checkpoints are accepted by the explicit legacy loader but are not relabeled as trained latent heads.

### Calibrated shift/conditional variability diagnostic

Across five deterministic experiment seeds, the fixed-feature nearest-reference detector, trained on 100 normal images per seed (50 fit and 50 disjoint calibration), had a mean in-distribution false-alarm rate of **4.6%** over 100 fresh normal images per seed and detected **100%** of 100 shifted images per seed (a deliberately large red-channel change). Repeated deterministic outcomes had zero sample conditional variance, while repeated randomized outcomes had positive variance. The report is `sample_outputs/uncertainty_diagnostics.json`. Conditional variance can reflect unobserved confounders and calibration coverage relies on stable distributional assumptions.

The agent now calibrates this detector once from its first 60 *observed training* transitions, records the fit with source evidence references in the ledger, and consults the detector in the planner. This is an implementable bounded anomaly alarm, not a generally calibrated probability of correctness.


### Controlled continual-learning retention

A five-seed matched-update contextual visual-rule benchmark now shows observable forgetting and its reduction through experience replay. Each arm used 80 initial training and 80 adaptation updates with batch size 16 and matched initial parameters per seed. Mean old-task image MSE after stage 1 was 0.008849 for both arms. After rule change it rose to **0.031491 without rehearsal**, compared with **0.008988 with a 50/50 old/new rehearsal batch**. New-task final MSE was 0.009347 without rehearsal and 0.009950 with rehearsal. Dormant latent fractions, measured wall time and buffer byte costs were also recorded for each run in `sample_outputs/continual_rule_shift_report.json`.

This now demonstrates forgetting reduction in a controlled source-of-change experiment; unlike the earlier inconclusive retention diagnostic, the old-task deterioration is genuine and measured. It does **not** demonstrate retention under full procedural-world changes, broad transfer or optimal replay scheduling. The Section 14 engineering acceptance gate is therefore marked **verified locally in its controlled setting**, not generally solved.

### Reproducible CLI and resumable temporal optimizer

`pytest` and `python -m pytest` both work from a fresh source directory, with local `pythonpath` specified in `pyproject.toml`. The temporal v3 checkpoint includes image/latent head parameters, update count, mode, and optimizer moments; an exact continuation test compares resumed against uninterrupted training. The local suite passes **69 tests** on Python 3.13, while full dependency installation under Python 3.11+ and independent verifier isolation remain outstanding.

### Visual curiosity continuation — 76-test local suite

The complete five-seed development and separate five-seed confirmation results for a newly integrated action-specific validated-progress policy, matched against random and prediction-error action selection, are reported in [`section8_exploration.md`](section8_exploration.md) and the JSON records under `sample_outputs/`. The TV effect uses a *separate sixth no-op action* and is not confounded with legitimate door/switch interaction. In the fresh confirmation set, the progress policy selected TV on 2.08% of final-half decisions against random's 17.44%, and achieved 98.8 average episode-position combinations against 32.4 for raw prediction error. Random achieved more coverage (104.4) and a lower common held-out prediction error. **This is negative evidence for a general sample-efficiency advantage.** No reliable goal completion was observed, and Section 08 remains partial.

### Section 08 v4 — visual-frontier physical coverage on fresh seeds (85-test suite)

A local, image-only visual-odometry/frontier controller was added alongside the previous three curiosity baselines. It uses explicit camera-shift and movement-action priors, and tracks both freely chosen and externally selected actions without consulting gridworld diagnostic state. See `docs/section8_visual_frontier.md` for the independent final results and test procedure.

Ten fresh evaluation seeds 30–39 (500 interactions per policy/seed/task) yielded mean **physical position coverage** key 34.0 vs random 21.9, switch 35.2 vs 23.6, and noisy TV 36.4 vs 21.6. Corresponding paired 95% Student-t coverage difference intervals were [+9.63,+14.57], [+8.21,+14.99], and [+10.78,+18.82] positions. Two task wins occurred across visual-frontier's 30 runs, versus zero for random, too few to justify a task success claim. Held-out feature MSE was worse for the TV task (0.002962 vs 0.002565 for random). The benchmark uses exploratory, non-multiplicity-adjusted intervals and unsealed local seeds.

All raw seed rows, including the previous policy's development/early-confirmation seeds 0–4/20–29 and the final policy's untouched 30–39 seeds, are stored separately in `sample_outputs/`. The paired verifier rejects missing, duplicate and unequal-budget candidate comparisons. The reported result is improved observable exploration, **not** demonstrated efficient learning, reliable control, or recursion.

## Incremental acceptance check — Section 02, verifier evidence integrity (2026-11-16)

The local verifier promotion policy now rejects empty or unequal paired vectors, NaN/Infinity, non-numeric and boolean entries, negative prediction losses, and gains inconsistent with the corresponding measured losses. A new adversarial regression test covers seven malformed-evidence classes plus a valid positive case. The complete local suite passed **86 tests** with `python -m pytest -q` (Python environment of this run). This is an integrity improvement, **not** OS isolation or an independently protected evaluation. The Section 02 acceptance gate remains partial.

## Section 02 v6 — strict verifier request boundary

The worker previously coerced fractional, string and boolean budgets through `int()` before validation; direct budget checks also accepted fractional values. Requests now require exact integer budgets, reject duplicate JSON keys and non-finite JSON constants, cap input at 65,536 bytes, and validate version identifier types and lengths. Two new regression tests exercise the parser and real subprocess rejection; the full local suite passed 88 tests. Reproduce with `python -m pytest -q`. No promotion threshold was changed and no candidate was promoted. These are request-integrity controls, not OS isolation, secret evaluation, or a physical compute-resource cap. Section 02 remains partial.

## Section 02 v7 — kernel resource containment (2026-11-16)

The worker installs fixed POSIX hard and soft limits before parsing or importing the evaluator: 600 CPU seconds, 16 MiB per regular file, 256 descriptors and zero core-dump bytes. Existing stricter limits are preserved. Setup errors abort evaluation; unsupported platforms fail closed. The existing parent wall-time timeout and logical budgets remain unchanged. Four new tests verify actual kernel file-size denial, retained descriptor limits and inability to raise the tested hard limit, CPU termination under an inherited one-second cap, refusal when setup fails, and a valid five-step one-replicate subprocess evaluation that remains ineligible for promotion. Full suite: 92 passed in 20.04 seconds (Python 3.13.5, Linux, torch 2.10.0+cpu). Reproduce with `python -m pytest -o addopts='' -q`. See `sample_outputs/section02_v7_pytest.txt`.

This is per-process resource containment, not a hostile-code sandbox; limits do not cap aggregate descendants, memory, total storage or pipe output, and do not isolate the filesystem or network. Privileged processes may be capable of raising hard limits. The fixed protected seed source is still visible locally. Section 02 and 17 remain partial; no candidate was promoted. Next gate: separately credentialed, read-only evaluator with private assets and denied network/filesystem access, verified by negative probes.


## v8 incremental verification — version-file path confinement (2026-11-16)

Starting from v7 commit `61522be`, inspection identified that `VersionManager.rollback()` and persisted `ACTIVE.json` could accept path-component version identifiers; promotion had a separate, incomplete identifier check. The revision introduces one strict ASCII identifier validator (`[A-Za-z0-9][A-Za-z0-9_-]{0,127}`) at every version-file boundary: writing, activation/rollback, active-version reading and promotion. It rejects traversal, separators, control characters, invalid types and oversized names.

**Executed evidence:** `python -m pytest tests/test_rollback.py -q` passed 4 tests; `python -m pytest -q` passed 94 tests (no failures). Two new tests cover invalid identifier inputs, tampered ACTIVE.json and a policy-approved but unsafe candidate identifier. This is a local file-path validation improvement, **not** filesystem or network isolation, verifier-private assets, or independently validated learning gain. No model promotion occurred.


## v9 incremental verification — kernel network syscall denial (2026-11-17)

Starting from v8 commit `5445269` (descendant of `f582278`), the normal verifier worker now installs a fixed Linux libseccomp filter before parsing requests or importing the evaluator. Socket creation (including Unix sockets), selected network operations and io_uring syscalls return EPERM. Filter initialization failure aborts the worker; no permissive fallback exists. Linux with libseccomp.so.2 and kernel seccomp support is now required for worker evaluation.

Executed `python -m pytest tests/test_verifier.py -o addopts='' -q`: 12 passed in 10.84s. Executed `python -m pytest -o addopts='' -q`: 96 passed in 23.26s; raw output in `sample_outputs/section02_v9_pytest.txt`. New tests exercise real IPv4/IPv6/Unix socket denial and inheritance across exec, plus fail-closed setup. Existing legitimate five-step evaluation remains operational and ineligible for promotion.

This is a syscall-denial layer, not complete network or filesystem isolation: pre-opened descriptors, filesystem-mediated communication, untrusted local processes and other IPC remain outside the guarantee. No independently credentialed verifier, private seed assets, aggregate limits or general learning gain were established. Sections 02 and 17 remain partial; no candidate was promoted.


## Incremental verifier hardening — v10 (2026-11-17)

The Linux seccomp filter now additionally rejects unlink, rename, mkdir, link,
symlink, truncate, chmod and chown syscall families, including common *at variants.
The new subprocess test verifies EPERM for unlink/rename/mkdir/chmod/truncate,
checks descendant inheritance, and verifies the original sentinel is unchanged.
An additional fail-closed regression test verifies that failed filter installation
prevents request evaluation. The existing legitimate verifier evaluation test also
passes with the expanded filter. Full suite: **98 passed** (`python -m pytest -q`).

**Security boundary:** This is only syscall-level defense in depth, not filesystem
confinement. open/openat with write flags, inherited writable descriptors,
mmap-based writes and several other mutation channels remain possible. There is
no independently credentialed worker or private evaluation data store. No
model candidate was promoted and no new learning-performance claim is made.

## Continuation v11 — inherited descriptor boundary (2026-11-17)

Source baseline: v10 commit `f1734d5` (descendant of `f582278`). The worker now enumerates `/proc/self/fd` before request parsing, closes all descriptors above standard input/output/error, and aborts if descriptor enumeration fails or a live descriptor cannot be closed. This addresses inherited file-handle exposure, including descriptor numbers above the verifier's eventual soft descriptor limit. The new subprocess test deliberately passes a read-capable secret-file descriptor at fd 220 and verifies that it becomes invalid while standard streams remain functional. A second test verifies fail-closed behavior when `/proc/self/fd` cannot be read. Existing in-process worker failure tests mock descriptor closure to avoid closing pytest's own capture descriptors.

`python -m pytest -q`: **100 passed** locally (Python 3.13, Linux). This is a narrow kernel/process-boundary test, not a hostile-code isolation proof. The verifier still can open readable paths by name, and `open/openat` writes, mmap, inherited memory, IPC, and evaluator-secret separation remain unresolved. No candidate promotion or research improvement is claimed.

## Continuation v12 — request lineage validation (2026-11-17)

The worker previously accepted candidate and parent strings containing path separators, even though the version manager rejected them. The request boundary now reuses `validate_version` and rejects explicit mismatches between the modification's parent version and the envelope parent. A new regression test covers traversal strings, control characters, oversized names, invalid types and mismatched parent identifiers.

`python -m pytest -q` completed with **101 passing tests** in this environment. A pre-change baseline run of `tests/test_verifier.py` had **one failure**: the legitimate worker evaluation could not create a temporary directory under the seccomp policy. The full post-change suite passed; the intermittent/environment-dependent baseline failure needs isolated repeat testing before claiming that legitimate evaluations are reliably supported. This change does not establish filesystem isolation or private evaluation assets. No promotion was attempted.

## Continuation v13 — seccomp/tempfile incompatibility reproduced (2026-11-17)

The v12 verifier-only baseline failure was reproduced deterministically: `test_valid_worker_request_under_kernel_limits` failed because Python could not create a temporary directory after the inherited seccomp filter denied `mkdir`/`mkdirat`. This is a **functional regression** of the security hardening, not an intermittent success. The worker now performs a temporary-directory capability preflight after input validation and before evaluator imports, failing closed with an explicit diagnostic rather than proceeding into an opaque dependency failure. Two new tests cover real kernel enforcement and preflight ordering. The formerly optimistic legitimate-evaluation test now asserts the actual fail-closed behavior; it must be changed back to a successful evaluation requirement after a safe redesign.

`python -m pytest -q -o addopts=''`: **103 passed** in 31.34 seconds on this host. This is **not** proof of a working evaluator; legitimate promotion evaluations are currently unavailable. No syscall restrictions were relaxed, no model promotion occurred, and Section 02 remains partial. The next fix requires a properly isolated writable scratch filesystem or separate-credential sandbox where temp directory creation is allowed without exposing verifier-private assets.

## Continuation v14 — bounded JSON nesting (2026-11-17)

Starting from v13 commit `3eb23aa`, the verifier request boundary now rejects JSON object/list nesting deeper than 32 levels after parsing and before typed evaluator construction. This bounds post-parse traversal depth; the existing 64 KiB byte cap remains. Two new tests cover a deeply nested request through the parser and through the actual verifier subprocess, and a shallow valid request. Full suite: **105 passed in 30.26s**, `python -m pytest -q -o addopts=''`; raw output in `sample_outputs/section02_v14_pytest.txt`. The policy was not relaxed and no candidate was promoted. This does not repair the confirmed seccomp/tempfile regression, establish filesystem confinement, or validate general learning improvement. Section 02 remains partial.

## Continuation v15 — process-memory and handle syscall denial (2026-11-17)

From v14 commit `675211f`, the inherited Linux seccomp policy now rejects `ptrace`, `process_vm_readv`, `process_vm_writev`, `pidfd_getfd`, and `open_by_handle_at`. A new subprocess regression invokes each actual libc syscall and confirms kernel EPERM rather than relying on source inspection. `python -m pytest -q -o addopts=''` passed **106 tests in 30.15s**; raw output: `sample_outputs/section02_v15_pytest.txt`. This is defense in depth, not an independent credential boundary. Legitimate verifier evaluations remain blocked by the existing mkdir/seccomp incompatibility; no promotion or general learning gain is claimed. Section 02 remains partial.

## Continuation v16 — typed request envelope validation (2026-11-17)

From v15 commit `2e7980e`, the verifier request parser now rejects non-object configurations, missing or extra modification fields, non-object modification parameters, non-string target/operation and unsafe modification parent identifiers before evaluator imports. Two new regression tests cover malformed envelopes and subprocess rejection. `python -m pytest -q -o addopts='' tests/test_verifier.py`: **24 passed in 20.29 seconds**. A complete suite run reached its 108 test progress markers but exceeded the command timeout before emitting a final summary; **full-suite pass is not claimed**. Existing seccomp `mkdir` incompatibility remains, so legitimate evaluations are still blocked; no promotion was attempted.

## Continuation v17 — JSON exponent overflow rejection (2026-11-17)

Starting from v16 commit `4f12043`, the protected worker now uses a finite-only `parse_float` hook. Previously `json.loads` accepted a syntactically valid `1e999` number and silently converted it to Python infinity, bypassing the `parse_constant` check that catches literal `Infinity` and `NaN`. Two tests cover the parser and an actual verifier subprocess. Verifier suite: **26 passed in 23.06s** (`python -m pytest -q -o addopts='' tests/test_verifier.py`). The complete suite displayed all test progress markers but did not exit before the 100-second timeout; therefore a full-suite pass is **not** claimed. Raw outputs are stored in `sample_outputs/section02_v17_*.txt`. The seccomp temporary-directory regression is unresolved; legitimate verifier evaluations remain blocked. No candidate promotion or general learning improvement was attempted.

## Continuation v18 — bounded JSON node traversal (2026-11-17)

Starting from v17 commit `5371384`, the verifier now rejects JSON trees with more than 8192 nodes, in addition to its existing 32-level depth and 65536-byte request limits. A shallow request with 8193 list elements fits within the byte limit and previously passed the depth check. Two new regression tests cover parser and actual verifier subprocess rejection. `python -m pytest -q -o addopts='' tests/test_verifier.py`: **28 passed in 26.21s**; raw output in `sample_outputs/section02_v18_verifier.txt`. A full-suite attempt did not produce a completion summary within the execution budget; the captured incomplete output is in `sample_outputs/section02_v18_full_incomplete.txt`. No full-suite pass is claimed. Existing seccomp temporary-directory incompatibility and lack of independent filesystem isolation remain; no promotion or general learning improvement occurred.


## v19 incremental verifier lineage check — 2026-11-17

The protected worker now rejects a candidate version identical to its parent before evaluator imports. This prevents an ambiguous self-comparison from being interpreted as a new lineage generation. Two tests exercise direct parsing and the real worker subprocess. `python -m pytest -q tests/test_verifier.py` completed with **30 passed** (exit status 0); raw output is in `verification/v19_verifier_pytest.txt`. A preceding full-suite attempt did **not** complete within the execution window; its partial progress is not a pass. The seccomp temporary-directory incompatibility remains unresolved, and no candidate was promoted. This is an input-integrity improvement, not demonstrated learning capability.

## v20 incremental verifier wire-encoding validation — 2026-11-17

Starting from v19 commit `edd4926`, the worker now explicitly decodes the request as strict UTF-8 and rejects UTF-8 BOMs. Previously `json.loads(bytes)` automatically detected UTF-16/UTF-32, allowing alternate wire encodings despite the intended JSON-over-UTF-8 boundary. The change is confined to input parsing and does not relax protected syscall restrictions. Two tests verify UTF-8 acceptance and UTF-16/UTF-32/BOM/invalid-byte rejection, including a real worker subprocess. The verifier-specific pytest output reports **32 passed in 30.84s** in `verification/v20_verifier_pytest.txt`; the encompassing command was externally timed out after pytest printed its final summary, so shell completion for that invocation was not observed. An independent focused invocation completed with **2 passed, 30 deselected in 3.71s**. No full-suite pass is claimed. The existing seccomp/temp-directory incompatibility, filesystem isolation and verifier-private asset gaps remain. No candidate promotion or general-learning improvement occurred.

## v21 — Unicode surrogate rejection (2026-11-17)

From v20 commit `33b25a1`, `validate_json_depth` now rejects unpaired Unicode surrogate code points in JSON strings, including object keys, before typed evaluator construction. Python's JSON decoder otherwise accepts escaped lone surrogates, which cannot be faithfully re-encoded as strict UTF-8. The traversal also counts dictionary keys toward the existing 8192-node complexity budget. Focused test command `python -m pytest -q -o addopts='' tests/test_verifier.py -k surrogate` exited successfully: **2 passed, 32 deselected in 7.57s**. A verifier-wide run did not complete within the available execution window; its partial log is `verification/v21_verifier_pytest.txt` and is not a pass. No syscall policy changed, no promotion occurred, and legitimate evaluation remains blocked by seccomp mkdir restrictions.

## v22 — legacy creat(2) syscall denial (2026-11-17)

Starting from v21 commit `4cb2847`, the Linux seccomp filter now denies the legacy `creat` syscall, which can create or truncate a file independently of the already-denied `truncate` syscall. A new real-kernel subprocess test invokes libc `creat` against an existing sentinel and verifies `EPERM` and unchanged contents. Focused run: **1 passed, 34 deselected in 2.06s**; verifier suite printed **35 passed in 32.25s**, recorded in `verification/v22_verifier.txt`. The outer tool reported a timeout despite pytest printing its completed summary; do not treat this as a confirmed clean shell exit. No full-suite pass is claimed. This is only a narrow syscall reduction: `open/openat` writes and the legitimate tempfile creation failure remain. No model improvement or candidate promotion occurred.

## v23 — write-capable open/openat syscall flag denial (2026-11-17)

Starting from v22 commit `5a4e4d6`, the inherited Linux seccomp policy now rejects write-capable `open` and `openat` flags (`O_WRONLY`, `O_RDWR`, `O_CREAT`, `O_TRUNC`, `O_APPEND`, `O_TMPFILE`) using libseccomp masked-argument comparisons. A real subprocess regression demonstrates EPERM on each tested write variant while ordinary read-only open succeeds and a sentinel remains unchanged. Focused test completed with **1 passed, 35 deselected** (exit 0), recorded in `verification/v23_focus.txt`. The verifier suite printed **36 passed in 33.08s** in `verification/v23_verifier.txt`; the outer execution tool timed out after the summary, so clean shell completion is not confirmed. This is defense in depth, not complete filesystem confinement: memory mapping, alternative write syscalls and directory-creation incompatibility remain. No model promotion or general learning claim.

## v24 — anonymous file creation denial (2026-11-17)

From v23 commit `edbcac1`, the Linux verifier seccomp policy now denies `memfd_create`, preventing candidate code from creating new anonymous writable memory-backed files via that syscall. A real subprocess regression invoked `os.memfd_create` after installing the filter and observed `EPERM`. Focused pytest completed with **1 passed, 36 deselected** (exit 0); raw evidence is `verification/v24_focus.txt`. Verifier-wide pytest output reports **37 passed in 32.71s** (`verification/v24_verifier.txt`); the outer execution command timed out, so a clean enclosing shell exit was not independently confirmed. No full-project suite pass is claimed. An exploratory attempt to filter writable shared `mmap` was **not validated** and was reverted before commit; no mmap protection is claimed. Legitimate evaluation remains blocked by mkdir restrictions; no candidate was promoted. Section 02 remains partial.


## v25 — openat2 filesystem policy coverage (2026-11-17)

From v24 commit `b8a0210`, the policy denies `openat2` entirely because its flags reside in a pointer-referenced structure, not a directly filterable syscall argument. A real subprocess regression first failed on the old policy (write-capable descriptor returned), then passed after the denial, together with the existing open/openat regression: **2 passed, 36 deselected**, exit 0. Sentinel contents were preserved and ordinary read-only access remained functional. Commands: `python -m pytest -q -o addopts='' tests/test_verifier.py -k openat2` (before), then `python -m pytest -q -o addopts='' tests/test_verifier.py -k 'openat2 or write_capable_open'` (after). Logs: `verification/v25_before.txt`, `verification/v25_focus.txt`, `verification/v25_verifier.txt`. The verifier-wide invocation exceeded the external execution deadline; no clean exit is claimed. Compilation passed. No policy relaxation, model promotion, or general learning gain occurred. Section 02 remains partial: legitimate temporary directory creation and independent filesystem isolation remain unresolved.

## v26 — extended attribute mutation protection (2026-11-17)

From v25 commit `493ec9c`, a real subprocess regression reproduced permitted `setxattr` mutation before the fix (1 failed, exit 1). The policy now denies setxattr/lsetxattr/fsetxattr and removexattr/lremovexattr/fremovexattr. The regression directly exercises path-based set/remove and verifies preserved metadata and file contents; the other variants are not independently exercised. Focused command `python -m pytest -q -o addopts='' tests/test_verifier.py -k 'xattr or openat2 or write_capable_open'` completed with 3 passed, 36 deselected, exit 0. Logs: `verification/v26_before.txt`, `verification/v26_focus.txt`. The verifier-wide command timed out at the outer execution deadline; raw output in `verification/v26_verifier.txt`; no clean exit or full-project pass claimed. Compilation passed. No policy relaxation or candidate promotion. Section 02 remains partial; legitimate temporary-directory creation and independent filesystem isolation remain unresolved.

## v27 — runner response identity and acceptance validation (2026-11-17)

Starting from v26 commit `698b1d4`, the caller now rejects successful-looking responses with mismatched candidate/baseline versions, non-boolean acceptance, or inconsistent/unrecognized promotion reasons. A mocked subprocess response regression failed before the correction (1 failed, exit 1), then passed with a positive matching-report control. This is a controlled protocol test, not evidence of a compromised real verifier or successful learning. Focused command `python -m pytest -q -o addopts='' tests/test_verifier.py -k 'invalid_success_identity or valid_worker_request or promotion'`: 4 passed, 36 deselected, exit 0. Full verifier suite `python -m pytest -q -o addopts='' tests/test_verifier.py`: 40 passed in 32.66s, exit 0. Python compilation passed. Logs: `verification/v27_before.txt`, `verification/v27_focus.txt`, `verification/v27_verifier.txt`. A separate namespace feasibility probe (`unshare -Urnm sh -c 'id; mount -t tmpfs tmpfs /mnt; echo mount_exit=$?'`) is recorded in `verification/v27_namespace_probe.txt`; it does not establish filesystem isolation or permit weakening the active policy. Section 02 remains partial; legitimate evaluation still fails closed. No full-project test run or candidate promotion occurred.

## v28 — caller evidence and budget consistency (2026-11-17)

Starting from `d8eef85`, a mocked successful response containing an incorrect evaluation budget was accepted before the fix (1 failed, exit 1). The runner now checks exact requested training cost and replicate count, finite numeric metrics, nonnegative losses and regression, arithmetic consistency of mean gain, ordered confidence bounds and acceptance/reason consistency against the existing protected thresholds. Eight malformed response variants and a matching positive control are exercised. This cannot authenticate underlying measurements or reconstruct the bootstrap from summary statistics. Commands: `python -m pytest -q -o addopts='' tests/test_verifier.py -k inconsistent_report_budget` (before); `python -m pytest -q -o addopts='' tests/test_verifier.py -k 'inconsistent_report_budget or invalid_success_identity'` (2 passed, exit 0); `python -m pytest -q -o addopts='' tests/test_verifier.py` (41 passed in 32.22s, exit 0); `python -m compileall -q aura protected_verifier tests` (exit 0). Raw logs are in `verification/v28_*.txt`. Namespace support was observed again but no isolated writable evaluator was implemented. The active syscall policy and protected thresholds are unchanged. Section 02 remains partial; no full-project suite run, real successful learning evaluation or promotion is claimed.

## v29 — unambiguous verifier response framing (2026-11-17)

Starting from `461a1e3` (ancestry to `f582278` checked), the caller previously accepted duplicate JSON keys and selected the last stdout line. A mocked duplicate-key response reproduced acceptance before the fix (1 failed, exit 1). The caller now parses the entire stdout as one JSON document, rejects duplicate keys at every nesting level, nonstandard constants, non-object or malformed envelopes, extra envelope fields and success with nonzero exit. Eight malformed variants and a positive control are exercised. Focused command `python -m pytest -q -o addopts='' tests/test_verifier.py -k 'ambiguous_response_framing or inconsistent_report_budget or invalid_success_identity or valid_worker_request'`: 4 passed, 38 deselected, exit 0, including the real worker fail-closed regression. The verifier-wide command timed out with only partial progress; no pass is claimed. `python -m compileall -q aura protected_verifier tests` exited 0. Logs: `verification/v29_before.txt`, `verification/v29_focus.txt`, `verification/v29_verifier.txt`. The syscall policy and promotion thresholds are unchanged. Section 02 remains partial; isolated writable scratch, private assets and successful real evaluation remain unresolved. No candidate promotion or learning gain occurred. Section 01 remains an overarching dependent mission gate.

## v30 — bootstrap summary feasibility (2026-11-17)

From `b35d9e0`, a mocked report with confidence bounds exceeding the sum of all nonnegative baseline losses was accepted before the fix (1 failed, exit 1). The caller now bounds bootstrap means using total baseline and candidate losses (with floating-point tolerance), rejects numeric conversion overflow, and preserves the protected sentinel for fewer than three replicates. Focused command `python -m pytest -q -o addopts='' tests/test_verifier.py -k 'impossible_bootstrap_summary or small_sample_bootstrap_sentinel or inconsistent_report_budget or ambiguous_response_framing or invalid_success_identity'`: 5 passed, 39 deselected, exit 0. Compilation exited 0. The verifier-wide command exceeded the execution deadline; no clean exit or full-suite pass is claimed. Logs: `verification/v30_before.txt`, `verification/v30_focus.txt`, `verification/v30_verifier.txt`. This is a necessary summary consistency check, not authentication of raw measurements. Section 02 remains partial; the isolated writable scratch environment and real successful evaluation remain unresolved. No protected policy change or candidate promotion.

## v31 — isolated scratch feasibility integration test (2026-11-17)

From `99273e2`, the new standalone controlled probe creates user, mount and network namespaces, marks mount propagation private, mounts a 4 MiB tmpfs, changes root and cwd, drops effective/permitted/inheritable capabilities and sets no-new-privileges. It verifies scratch creation/readback, inability to open an external sentinel and denial of a second chroot; the parent checks its sentinel and mountpoint remain unchanged. `python scripts/probe_private_scratch.py` exited 0; `AURA_TEST_NAMESPACES=1 python -m pytest -q -o addopts='' tests/test_namespace_probe.py` passed 1 test (exit 0); compilation exited 0. Logs: `verification/v31_namespace.txt`, `verification/v31_focus.txt`, `verification/v31_compile.txt`. The test is opt-in and does not silently pass when enabled on an unsupported host. This is a trusted probe, not a hostile-code sandbox: no candidate execution, runtime dependency provisioning, PID isolation or independent assets are tested. The active verifier and promotion rules remain unchanged, and legitimate evaluation remains blocked. No full-suite rerun or candidate promotion. Section 02 remains partial; Section 01 depends on the other gates.

## v32 — PID namespace feasibility (2026-11-17)

From `922b115`, the standalone trusted scratch probe now includes a PID namespace (`unshare -Urnmp --fork --kill-child`), checks PID 1 and namespace identity differs from the parent before chroot, and checks parent PID 0 and absent `/proc` after chroot. Existing scratch and host sentinel checks remain. The new test initially failed because evidence was missing (1 failed, 1 passed). A private read-only procfs mount experiment failed with kernel `VFS: Mount too revealing` (2 failed); that mount attempt was removed, not bypassed. Final `AURA_TEST_NAMESPACES=1 python -m pytest -q -o addopts='' tests/test_namespace_probe.py` passed 2 tests, exit 0; `python scripts/probe_private_scratch.py` and compilation exited 0. Logs: `verification/v32_*.txt`. This is not a hostile-code sandbox or a working evaluator: active worker descriptor enumeration requires `/proc/self/fd`; a supported private procfs or separately validated descriptor-isolation design and dependency provisioning are still prerequisites. Protected verifier code and promotion rules unchanged; no full suite or learning gain claimed. Section 02 remains partial; Section 01 remains dependent.

## v33 — procfs-independent descriptor closure (2026-11-18)

From `bc4f8ca`, the worker now prefers libc `close_range(3, UINT_MAX, 0)` before request parsing, retaining the prior procfs inventory fallback on unavailable or failed kernel closure. The real subprocess regression passes fd 220, lowers its soft RLIMIT_NOFILE to 64 and simulates missing procfs enumeration; before the change it failed (1 failed, exit 1), afterward it closes fd 220 and preserves stdio. Separate regressions exercise fallback and fail-closed behavior when both mechanisms fail. `python -m pytest -q -o addopts='' tests/test_verifier.py -k 'inherited_high or descriptor_inventory or descriptor_procfs'`: 3 passed, 42 deselected, exit 0. Compilation exited 0. The verifier-wide invocation exceeded the execution deadline; no clean exit claimed. Logs: `verification/v33_*.txt`. This is not a test inside a chroot; actual namespace integration and legitimate learning evaluation remain pending. No seccomp or promotion threshold changes, candidate promotion, or general learning gain. Section 02 remains partial; Section 01 remains dependent.

## v34 — real namespace descriptor closure (2026-11-18)

From `01cfdfe`, the trusted namespace probe now imports the actual worker closure before chroot, passes a real read-only sentinel descriptor numbered at least 220, verifies its contents before isolation, reduces the soft descriptor limit to at most 64 and invokes closure after chroot and capability drop with `/proc` absent. It checks EBADF and preservation of fd 0/1/2, then performs existing scratch and sentinel checks. The new acceptance test initially failed for missing evidence (not a reproduced security vulnerability). `AURA_TEST_NAMESPACES=1 python -m pytest -q -o addopts='' tests/test_namespace_probe.py` passed 3 tests in 6.09s, exit 0; `python -m pytest -q -o addopts='' tests/test_verifier.py -k 'inherited_high or descriptor_inventory or descriptor_procfs'` passed 3 tests in 5.19s, exit 0; standalone `python scripts/probe_private_scratch.py` and `python -m compileall -q aura protected_verifier scripts tests` exited 0. Logs: `verification/v34_*.txt`. Runtime imports occur before chroot, so this does not establish dependency provisioning inside isolation, candidate execution, or complete sandbox confinement. Protected worker and thresholds unchanged. No full suite run, candidate promotion, or learning gain claimed. Section 02 remains partial; Section 01 remains dependent.

## v35 — read-only standard-library provisioning (2026-11-18)

From `3b830a7`, the opt-in trusted namespace probe bind mounts the Python standard library read-only with nosuid,nodev,noexec before chroot. After capability drop and actual worker descriptor closure, it restricts sys.path to `/runtime`, disables bytecode writes, checks colorsys was not previously imported, imports it from `/runtime/colorsys.py`, verifies RGB-to-HSV computation and requires EROFS on write-open of that file. Existing scratch and sentinel checks remain. The new regression initially failed for missing evidence (1 failed, exit 1); an intermediate fractions probe failed because the module was already loaded. Final `AURA_TEST_NAMESPACES=1 python -m pytest -q -o addopts='' tests/test_namespace_probe.py`: 4 passed in 8.35s, exit 0. Standalone `python scripts/probe_private_scratch.py` and compilation exited 0. Ancestry to f582278 verified. Logs: `verification/v35_*.txt`. This is a trusted pure-Python import microbenchmark, not a complete runtime or hostile-code sandbox; the interpreter and worker were loaded before chroot, native extensions and ML dependencies are untested, and host library exposure must be reviewed before production integration. Protected worker and thresholds unchanged; no full suite run or promotion. Section 02 remains partial and Section 01 dependent.

## v36 — breadth-first minimum architecture

From d3c1c0c, added a bounded non-deploying study and integrated smoke command, with a 31-section source coverage map. Focused command `python -m pytest -q -o addopts='' tests/test_minimum_study.py tests/test_archive.py tests/test_integration.py`: 4 passed in 2.18s. Real smoke `python -m scripts.minimum_architecture --output /tmp/aura36-smoke`: 20 interactions, valid ledger, checkpoint, zero completed episodes; protected verifier rejected temporary-directory capability, exit 2, no deployment. Full suite exceeded execution deadline; see partial output. No acceptance upgrade or independent learning gain claimed. Protected policy unchanged.

## v37 — separate online development validation and five-task integration

Parent `5f43cd0`; ancestry to `f582278` confirmed. The new real training regression reproduced an intersection between the old replay-derived validation bank and optimizer minibatches (before: 2 failed, exit 1, including one missing-report-field test). The agent now samples a dedicated eight-transition random-policy development rollout, hashes its data, retains fixed provenance in its ledger, charges a separate bounded budget and never adds this rollout to replay. If development budget is absent, the feedback signal is disabled. The noisy-task regression compares actual recorded training and validation transition bytes, and verifies repeated validation uses the same samples; identical observations across independently generated environments remain possible. Development seeds are deterministic, not a secrecy or mathematical noncollision guarantee. The fixed bank is reused adaptively and covers one task, so it is not independent test evidence. The resource report preserves the original training interaction count and adds both development and total environment interactions.

Focused command: `python -m pytest -q -o addopts='' tests/test_minimum_study.py tests/test_online_development_validation.py tests/test_integration.py tests/test_progress_integration.py tests/test_temporal_integration.py tests/test_resource_allocation.py tests/test_visual_frontier.py` — 18 passed in 4.65 s, exit 0. `python -m scripts.minimum_architecture --output NEW_DIRECTORY --all-tasks --steps-per-task 40` — 200 training + 40 development interactions, 10 completed episodes, zero successful goals, five valid ledgers, 430 optimization/concept steps (80 temporal updates included). Each family uses a fresh random initialization; no transfer is tested. Recursive study blocked by existing verifier temporary-directory capability failure, CLI exit 2; no deployment. Compilation passed. Raw before/focus/matrix logs in `verification/v37_*.txt`. Protected verifier files and promotion rules unchanged. No acceptance upgrade or independently established learning improvement. The extra development interaction cost must be included in future matched-budget comparisons.

Full regression command `python -m pytest -q -o addopts=''`: **135 passed, 4 skipped in 51.79 s**, exit 0 (`verification/v37_full.txt`). The skipped cases are opt-in namespace probes; they are not claimed as executed this revision. The matrix report contains original run-local paths; a separate archived matrix output preserves checkpoints and ledgers with relative paths. No protected verifier source changed.

## v38 — cross-section integration and reward-grounded behavior reuse

Parent `af3c192`. Five new integration regressions failed on the parent (task-local validation/fresh credit, skill episode support, missing meta measurements, reopened attempts, observational causal label), then 25 focused cases passed. Two more live-continuation and actor-artifact cases failed before implementation and the 15-case focus passed after. Two additional scheduling/curriculum tests failed before changes; the 22-test focused invocation passed after. Final opt-in route and integration focus: 17 passed in 8.71 seconds. The original skill test was corrected to provide distinct synthetic episodes rather than duplicating the same evidence. Logs in `verification/v38_*.txt` retain failures and intermediate runs.

The adaptive agent's online development bank is now keyed by task, with up to 8 additional interactions per family, fixed data reused adaptively, task-local error histories and explicit paired before/after gains. Missing gain does not count as an observed zero; scheduling uses per-mode selection count to prevent phase locking. Modes are action-policy choices, not independently tested learning operators. Curriculum first covers unattempted tasks. The agent preserves live episode state/seed/skill queue/counters across cumulative calls and saves a standalone actor artifact. A CPU regression compares uninterrupted 24 steps with 5/13/24 cumulative calls and identical observed trajectory and actor state. No full restart support is asserted.

The frontier regression fixes clearing blocked edges without resetting attempts. Public exploratory frontier vs random diagnostic: five families, seeds 80–82, 320 interactions per run, episode cap 80; 6/61 vs 0/60 successful completed episodes, 4,800 interactions per policy. `verification/v38_frontier_diagnostic.json` records raw rows. This is not a v37-v38 matched attribution comparison.

New experimental `RewardRouteExplorer` stores a bounded candidate only following observed positive terminal reward, replays from matching start images, checks intervening image similarity and falls back to frontier upon mismatch. `python -m experiments.reward_route_study --seed-start 90 --seeds 3 --steps 640 --horizon 80 --output verification/v38_reward_routes.json`: 29/122 successful episodes versus 8/120 for frontier alone, 9,600 interactions per policy across five task families, 21 exact reproductions; zero wins in noise. Raw paired rows included. The separate experiment is a public development study with static layouts, no preregistered inference or independent promotion gate. Opt-in integration requires curiosity condition so meta-controller cannot interrupt a route. Real experience IDs are passed after ledger append. Mined skills require distinct successful episode IDs and use monotonic candidate IDs, but are not independently verified options.

`python -m scripts.cross_section_study --output NEW_DIRECTORY`: adaptive run 400 training + 40 development interactions across five families, 10 completed episodes, no successes, 874 optimization/concept steps (80 temporal and seven accepted local concept candidates); paired greedy actor vs random on five episodes per family, both zero wins before and after. Reward-route run (diagnostic seed 1082, selected after prior exploratory results) 640 training + 8 development interactions in key family: 28/29 successful episodes, 26 exact successful route reproductions, two retained discovered routes, 97 skill candidates, 1,498 optimization/concept steps including 232 temporal updates; all retained discovery route references resolve; both ledgers verified. `python -m scripts.evaluate_control ...` using restored actor/world matched in-memory control rows exactly. Raw reports and logs included; separate evidence archive holds checkpoint and ledger data. Route control is not the neural actor. No general transfer claim.

Final default regression `python -m pytest -q -o addopts=''`: 149 passed, 4 skipped in 58.95 seconds, exit 0. Separate `AURA_TEST_NAMESPACES=1 python -m pytest -q -o addopts='' tests/test_namespace_probe.py`: 4 passed in 8.81 seconds, exit 0. Compilation exited 0. Real `python -m scripts.minimum_architecture --output NEW_DIRECTORY`: protected recursive study BLOCKED at study-1 (temporary-directory capability unavailable), exit 2, not deployed. No protected verifier source changed or candidate promoted. Acceptance status remains 4 locally verified, 23 partial, 3 open, 1 ongoing, and all limitations stated in the section register.

## v39 — narrowly provisioned native runtime

Parent `450b429`; original ancestry `f582278` verified. Section 02 trusted namespace experiment now mounts only the SQLite extension and its system shared library as separate read-only executable files; standard-library mount stays noexec. After chroot, capability drop and actual descriptor closure, verifies neither SQLite module was preloaded, imports from the explicit mount, evaluates SELECT 6*7 in memory and requires EROFS on both mounted files. New acceptance test initially failed (missing evidence key, not a demonstrated vulnerability); five opt-in namespace tests subsequently passed in 10.66s, exit 0. Standalone probe and compilation exited 0. Commands: `AURA_TEST_NAMESPACES=1 python -m pytest -q -o addopts='' tests/test_namespace_probe.py`; `python scripts/probe_private_scratch.py`; `python -m compileall -q aura protected_verifier scripts tests`. Logs `verification/v39_*.txt`. The library path assumes Debian-style multiarch layout; unsupported hosts fail rather than claim success. Interpreter and existing dependencies are preloaded; this is trusted code, not candidate confinement, ML computation or independent verification. No full suite run or promotion. Protected verifier unchanged; acceptance statuses unchanged.

## v40 — neural distillation of autonomously acquired successful trajectories

Parent `c7891e0`, preserving v39 and v38 history. `aura/planning/distillation.py` reconstructs observed experiences only from a verified local hash chain; successful terminal trajectories must have contiguous observations, consistent episode context/seed and no earlier termination. Nonpositive/truncated/incomplete episodes are excluded. The explicit recurrent image controller is randomly initialized, uses images and previous actions (no simulator diagnostics, task coordinates, pretrained weights, or route table) and trains at fixed update count with padded-token masking. Train/validation split is grouped by (context, environment seed), evidence IDs cannot overlap, and validation does not choose weights. Standalone safe weights-only checkpoint load; recurrent state resets each episode. The procedure assumes a closed, committed SQLite snapshot; hash chain is local provenance, not external attestation. The original v38 acquisition was not rerun; the archived ledger is the shared input. This is a manually invoked offline development study, not an autonomous promotion or integration with the running agent's DQN controller.

Reproduction from repository root: `unzip -p aura_omega_v38_learning_evidence.zip route/evidence.sqlite > /tmp/route.sqlite` (archive path may be absolute); for each seed 41, 42, 43 run `python -m experiments.neural_transfer_study --ledger /tmp/route.sqlite --output NEW_DIRECTORY --seed SEED --updates 120 --episodes 10`. The separate v40 evidence ZIP contains the closed source snapshot, three trained checkpoints and raw reports so the prior archive is not required. Seed 41 used 21/7 rewarded episodes and 419/140 transitions, validation accuracy 33.57% to 99.29% and loss 1.568616 to 0.022638; its initial run was extended with visual ablations and repeated at the same seed. Seeds 42 and 43 achieved 100% and 93.53% held-out action accuracy respectively. Across the three models (same data source, distinct deterministic partitions and initialization/rollout seeds), live goal counts 30/30 original size 9 at .05 corruption, 28/30 size 9 at .15, 7/30 size 9 at .50, 5/30 size 11 at .05; random 1/120, untrained 0/120, frozen initial image 4/120, blank image 0/120. All checkpoints reproduced corresponding live rollout rows on reload and source snapshot digests were unchanged. Evaluation seeds do not overlap source experience seeds (explicit collision check); same underlying fixed layout is not independent task generalization. Later seed replications and ablations are exploratory, not preregistered. All policies have equal 80-step episode caps; actual interactions differ with early completion. Training data acquisition and update costs are additional and not an equal total-resource comparison to random. No claim of generalizable skills or independent recursive improvement.

Focused command `python -m pytest -q -o addopts='' tests/test_distillation.py`: 3 passed in 1.92 s, exit 0. Tests cover observed/positive episode selection and chain integrity, seed group separation, overlap rejection, optimizer budget, training loss and checkpoint behavior. Full default suite timed out; partial output saved, no full result claimed. Protected verifier files unchanged; no candidate promotion and acceptance statuses unchanged. Real evaluation raw reports, logs and aggregate in verification/v40_*. The external verifier remains unvalidated for this pipeline.

## v41 — online shadow neural candidate generation

Parent `edcbda1`; preserves original `f582278` ancestry. New default-off `OnlineDistillation` is called at the end of non-random training episodes, after the final experience is committed. Task-local success counters trigger at the configured interval; two distinct seed groups are necessary. It samples at most the 32 most recent eligible episodes of at most 256 transitions, uses grouped held-out demonstrations, and checks observed evidence chain integrity. It starts a fresh CPU model in a forked Torch RNG context; matched public development seeds test the same initial model before and after optimization, are checked against recorded training seeds and previous shadow seeds, and training rejects a later collision. The scheduler caps candidates and updates and checks the worst-case rollout budget, charges actual shadow interactions, and adds those to total environment interactions. It does not use evaluation results to choose a candidate or alter the active exploration policy; no activation path or protected verifier source change. Failures may interrupt training; process restart, external evidence attestation, and long-ledger streaming are not solved. Later candidates may reuse earlier training or validation episodes; no claim of independent replications or successive generations.

Reproduce: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m scripts.online_imitation_study --output NEW_DIRECTORY` (seed 1082, chosen from prior diagnostic observations). One fresh integrated acquisition run: 640 training + 8 world-model development + 1,597 shadow rollout interactions = 2,245 total; 29 complete episodes, 28 successes; 1,738 optimization/concept steps including 240 shadow gradient updates. Candidate 1, at step 239, used six train/two validation episodes (119/40 transitions), action accuracy .05 -> .95 on validation, 0/8 -> 8/8 public goals (797 total rollout interactions). Candidate 2, at step 399, used 12 train/four validation episodes (240/79 transitions), validation accuracy .21519 -> .91139, 0/8 -> 8/8 goals (800 rollout interactions). Each uses distinct public seed set in the same key environment, .05 observation corruption, 80-step maximum. These are paired untrained-versus-trained evaluations, not equal total-resource policy comparisons or evidence of generalization to other layouts/tasks. The ledger contains 640 actual experience records and two candidate evidence rows; checkpoint digests and all references verified, both saved models reproduced the eight trained rows exactly. Reload diagnostic is external to training (317 more transitions), excluded from the training resource totals. Raw study in `verification/v41_study.json`; separate evidence archive contains source ledger, candidate models, and run artifacts.

Focused test run: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q -o addopts='' tests/test_online_distillation.py tests/test_distillation.py` -> 6 passed in 4.11s. Before implementation, the new test module failed collection because scheduler absent; not a behavioral failure. Full default suite with same thread caps: 155 passed, 5 opt-in namespace tests skipped in 57.15s, exit 0. No protected promotion, and section statuses unchanged (4 locally verified, 23 partial, 3 open, 1 ongoing).

Separate opt-in namespace command `AURA_TEST_NAMESPACES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q -o addopts='' tests/test_namespace_probe.py`: 5 passed in 10.24s, exit 0. These are trusted sandbox probes, not production verifier acceptance. Python compilation and `git diff --check` exited 0.

## v42 — parent-linked neural shadow generations

Parent `145b7dd` (original `f582278` ancestry). `imitation_inherit` is default false; when enabled, each task's next shadow candidate inherits the previous task-local checkpoint, whose filename, byte digest and ledger report are checked. Full eligible episodes must follow the parent candidate record in the ledger, with all ancestor train and validation seed groups excluded. Two distinct fresh seed groups are required; fixed episode-success trigger can include repeated seeds, but those are excluded from training/validation. Each candidate resets its optimizer, charges the same bounded update and matched before/after development rollout budgets, and records parent ID, digest, generation and baseline. No independent evaluator or active policy change. Sources are local tamper-evident evidence, not external attestation. The run is not process-restart resumable. Shadow candidate failures are not atomic across checkpoint and ledger writes; no recovery claim.

Reproduction: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m scripts.online_imitation_study --inherit --output NEW_DIRECTORY`; then `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m experiments.imitation_lineage_study --directory NEW_DIRECTORY --output NEW_DIRECTORY/lineage_diagnostic.json`. The latter is a post-training public diagnostic, not a selection criterion or a statistically independent protected confirmation. The acquisition seed 1082 was chosen in earlier diagnostics. One acquisition run yielded 640 training interactions, 8 model-development interactions, 1,437 shadow-evaluation interactions (2,085 total), 1,858 optimization/concept updates including 360 shadow updates, 28/29 successful exploration episodes, valid ledger. Candidates at steps 239,399,559 each used six training and two held-out demonstration seed groups; 119/40,120/40,120/40 training/validation transitions respectively; all groups disjoint across generations. Rollouts before→after 0/8→8/8, 8/8→8/8, 8/8→8/8. Validation cross-entropy 1st: 1.660588→.078086; 2nd: .025006→.001879; 3rd: .000603→.000032 (each paired within its own fresh held-out group, not across groups).

Common-seed comparison (ten fresh seeds per condition, same across generations, all distinct from acquisition and shadow development seeds): original size 9 at .05 noise 9/10,10/10,10/10; size 9 at .15 noise 9/10,10/10,10/10; size 9 at .50 noise 1/10,2/10,1/10; size 11 at .05 noise 0/10,2/10,0/10. Total 19/40→24/40→21/40; first successor improves descriptively, second regresses. No statistical significance claimed; one acquisition stream, sequential data and exploratory public diagnostic, budget not matched across generations. Additional diagnostic 5,767 interactions and checkpoint reload reproduction 477 interactions, separate from acquisition totals. All three checkpoints reproduced their recorded shadow after rows. No generational monotonicity or generalization established. No protected verifier sources changed or candidates promoted.

Two initial new tests failed because the configuration did not support inheritance (`verification/v42_before.txt`), not demonstrated behavioral bugs. After implementation eight focused tests passed, and the initial default suite had 157 passed, five opt-in tests skipped. After adding the diagnostic roundtrip/tamper regression, nine focused tests passed in 4.32s. Compilation and diff check passed. See final test logs and evidence archive. Acceptance statuses unchanged: four locally verified, 23 partial, three open, one ongoing.

Final full default suite: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q -o addopts=''` -> 158 passed, five namespace tests skipped in 61.42 s, exit 0. Separate opt-in `AURA_TEST_NAMESPACES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q -o addopts='' tests/test_namespace_probe.py` -> five passed in 10.93 s, exit 0. These namespace tests are trusted probes, not complete protected-verifier integration. Logs in `verification/v42_full.txt` and `verification/v42_namespace.txt`.

## v43 — bounded ancestral rehearsal and matched development comparison

Parent `4b6d098`; opt-in `imitation_rehearsal_episodes` defaults to zero. With inheritance, only complete successful episodes whose evidence IDs appeared in earlier training reports (never earlier validation seed groups) may be rehearsed. Recent distinct seed groups selected deterministically; at most 16 rehearsal and 32 total training episodes, 256 transitions per episode. Fresh groups still require post-parent evidence. Candidate reports distinguish fresh/rehearsal/validation groups and count full-batch transition presentations. The diagnostic validates ancestral provenance and replays each saved checkpoint. No protected verifier source changed.

The new regression failed on the parent because the option was missing (not a reproduced learning defect); final targeted command `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q -o addopts='' tests/test_online_distillation.py tests/test_distillation.py` passed 11 tests. Two real runs and six shadow checkpoint reloads succeeded. Reproduce in new directories (set OMP_NUM_THREADS=2 MKL_NUM_THREADS=2):

```
python -m scripts.online_imitation_study --output fresh --inherit
python -m scripts.online_imitation_study --output rehearsal --inherit --rehearsal 6
python -m experiments.imitation_lineage_study --directory fresh --output fresh/lineage_diagnostic.json --episodes 20
python -m experiments.imitation_lineage_study --directory rehearsal --output rehearsal/lineage_diagnostic.json --episodes 20
python -m experiments.rehearsal_comparison --fresh fresh --rehearsal rehearsal --output comparison.json
```

Comparison checks identical observed acquisition payloads and fresh split payloads (record IDs differ across runs), and matched case/seed tuples. Both runs complete 28/29 acquisition episodes successfully, 640 interactions, 360 shadow optimizer updates, and three 8/8 post-training shadow evaluations. Common diagnostics fresh 42/80, 44/80, 42/80; rehearsal 42/80, 43/80, 43/80. Rehearsal-only vs fresh-only discordance: 0/0, 3/4, 3/2. Equal aggregate goals, increased transition presentations 71,760 vs 43,080. Online total interactions 2,083 vs 2,085, diagnostic interactions 11,606 vs 11,585, checkpoint reload interactions 477 each. The evaluation includes public seeds previously used in v42. This is one selected acquisition configuration and not independent replications, not matched sample-processing compute, no independent promotion. No monotonic learning claim, restart not implemented. Logs `verification/v43_*`; evidence archive includes both runs and comparison JSON.

## v44 — budgeted whole-episode rehearsal

Parent `acea4fc`, original `f582278` ancestry. New default-off `imitation_budgeted_rehearsal` rotates complete fresh and eligible ancestral training episodes, always retaining at least one fresh episode. Each batch has at most the sum of fresh episode lengths in real transitions and at most fresh episode count × maximum fresh length in padded recurrent cells. No validation sequence participates in optimization. Per-episode presentation counts distinguish the available training pool from data actually presented; future rehearsal uses only the latter. Deterministic rotation is not learned prioritization. Full-pool pre/post measurement still costs extra; batch assembly, gradient cost, rollout and actual elapsed time are not bounded by these counters. The protected verifier was not changed. No activation or promotion.

Reproduce with `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m scripts.online_imitation_study --output NEW --inherit --rehearsal 6 --budgeted`, then `python -m experiments.imitation_lineage_study --directory NEW --output NEW/lineage_diagnostic.json --episodes 20` and `python -m experiments.rehearsal_comparison --fresh V43_EVIDENCE/fresh --rehearsal NEW --output comparison.json` with the same thread limits. The comparison resolves the recorded acquisition and split payloads and matches public scenario keys; these seeds are reused from v43, not independent evaluation. The budgeted run completed 28/29 acquisition episodes, 640 acquisition + 8 model-development + 1,435 shadow rollout interactions = 2,083; 360 shadow gradient updates; all three candidates 8/8 after-training shadow goals, all three restored checkpoints reproduce those rows. 11,755 diagnostic and 477 reload interactions are additional. Candidate optimizer transition counts 14,280 / 14,340 / 14,400; padded counts 14,400 each; rehearsal transition counts 0 / 7,140 / 7,200. Full-pool measurement padded counts 320 / 560 / 560 (forward only). Thus 43,020 actual optimizer transitions versus 43,080 fresh-only and 71,760 unbounded rehearsal, 40.05% fewer than unbounded. Common public goals 42/80, 40/80, 43/80 (125/240) vs fresh-only 42,44,42 (128/240) and unbounded 42,43,43 (128/240). No statistical superiority or matched total compute implied. No monotonic improvement, restart recovery or independent verification demonstrated.

Two new unit tests failed on the parent because the feature was absent (not a behavioral defect); final focused suite `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q -o addopts='' tests/test_distillation.py tests/test_online_distillation.py`: 14 passed in 4.22s, including integration and exclusion of unpresented ancestor evidence. Full default suite exceeded the tool execution window; partial log retained, no final result or exit status observed. Namespace tests not rerun. All 31 section statuses unchanged. Raw logs and matched summary in `verification/v44_*`; run evidence delivered separately.

## Section 09 primary gridworld acceptance

The v44 Section 09 continuation meets the local criterion for hypothesis and causal discovery. The [acceptance report](section09_acceptance.md) records the initial failure, fixed and fresh-seed results, data provenance, study assumptions, and reproduction commands. The current register is 5 locally verified, 22 partial, 3 open and 1 ongoing.
## v45 — equal-budget doorway variation (Section 01 prerequisite)

Parent `34959f7`, descended from `f582278`. Requested `aura_v43_layout_transfer.zip` was not present in mounted files or located by Library search; this is a separately implemented diagnostic, not a reproduction of that archive. Experiment-local world subclass alters only barrier doorway column; center matches base environment in regression test. Explicit privileged scripted teacher, not autonomous acquisition. Fixed trains at column 4; varied trains at 2,4,6; reserved 1,3,5,7 never used in training or validation. Five paired initialization seeds (41–45), 12 train + 3 validation episodes per arm, each 13 interactions, 120 updates, 18,720 actual and padded optimizer presentations per model. Same noise seeds and 60-step rollout caps; actual evaluation steps can differ. Per-column success out of 50 fixed: 0,0,0,50,0,2,36; varied: 0,31,0,40,0,50,50. Reserved totals 36/200 vs 50/200; center regresses 50/50 to 40/50; all reserved improvement comes from column 7, the other three positions remain unsolved. Public descriptive result, no independent validation or general mastery. Ten checkpoints reproduced all their evaluation rows after reload. Seven focused tests passed; full default suite timed out without final result. No protected verifier or production policy changes; no promotion. Reproduce: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m experiments.doorway_study --output NEW_DIRECTORY`. Raw report `verification/v45_doorway.json`.

## v46 — failure-driven doorway curriculum

Parent 3f75318, original f582278 ancestry verified. Three warmup demonstrations, 30 updates, five policy probes per training column, nine additional demonstrations, 90 updates. Uniform versus failure-weighted allocation with mandatory coverage; reserved columns excluded from allocation and validation. Five new initialization seeds 51–55, common public evaluation seeds 810000-series. Each arm/model: 195 teacher interactions, 120 updates, 15,210 optimizer transition and padded presentations; identical aggregate probe interactions (4,359 per arm). Optimizer resets at stage boundary in both arms. Only seed 53 changes allocation. Uniform vs adaptive per-column successes out of 50: [0,20,0,32,0,50,38] vs [0,19,0,38,0,50,36]; reserved 38/200 vs 36/200, overall 140/350 vs 143/350. No superiority or autonomous acquisition established. Ten checkpoints replayed exactly. Four focused tests passed; full suite attempt timed out without final result. Protected verifier unchanged; no promotion; acceptance statuses unchanged. Reproduce with `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m experiments.doorway_curriculum --output NEW_DIRECTORY`. Raw evidence `verification/v46_curriculum.json`. Original v43 diagnostic still not available in mounted files.

## Local integration of v46 and the focused continuation

Merged v46 (3762fc6) and the focused continuation (d26e1e9), which share base 34959f7. Source and tests merged cleanly. The two document conflicts were resolved by retaining the focused evidence and the v45/v46 neural studies, including the negative curriculum comparison. The v46 audit remains 4/31 for that revision. The integrated register carries 13 bounded local gates, including the controlled curriculum result. Section 13's separate v46 neural outcome is a regression on reserved layouts and remains unresolved.

On the integrated tree, 12 focused tests passed, compilation passed, and the uncertainty, movement transfer, and controlled curriculum experiments reproduced their saved JSON byte for byte. The full pytest suite could not start in this runtime because pytest is missing. These checks do not extend the scope of any prior acceptance gate.

## v47 stationary alignment
See section08_goal_acquisition.md and verification/section08/stationary_alignment.json. Original and revised public trials reproduced byte-for-byte. 12 focused tests and 5 subtests passed. No protected verifier changes or promotion; full suite not run. 13/31 local statuses preserved.

## v48 palette-masked alignment
320 matched development trials plus 320 fresh-seed trials; fresh 320 replayed byte-for-byte. Seven focused tests passed. Full suite not run. See section08_goal_acquisition.md and verification/section08/palette_*.json. No protected verifier changes; 13/31 retained.

## v49 calibration recovery
Parent 3032963, original ancestry checked. 320 matched public trials, exact replay, five focused tests passed. Clean 80/80 both; noisy 62/80 baseline versus 76/80 recovery. Full regression not run. Protected verifier unchanged; no promotion; 13/31 local gates retained. See section08_goal_acquisition.md.

### v50 passage recovery
640 matched public trials, fresh 320 rows replayed byte-for-byte, 11 focused tests passed. Clean 80/80 per arm per batch; noisy development 76/80 vs 78/80, fresh 74/80 vs 76/80. Four occlusion failures removed, six motion failures remain. Full suite not run. Section 08 partial; register unchanged (13/31). No protected promotion.

### v57 adaptive survey

636 matched trial records across development and fresh batches; fresh 420 records replayed byte-for-byte. 27 focused tests passed. Protected verifier unchanged compared with v56. No full regression run, Git commit, or default policy change. See `docs/v57_adaptive_routes.md` for scope and regressions.

### v58 early-key goal-only route

972 completed experimental records; 720 fresh records reproduced byte-for-byte in five shards. v57 vs v58 fresh 100/240 vs 90/240; v58 saves 1103 actions on 83 jointly successful trials, failure-penalized cost 12834 vs 11764. 30 focused tests passed. All 11 protected-verifier files unchanged vs v57. No Git commit, default policy change, full regression run or acceptance promotion. See `docs/v58_efficient_routes.md`.

### v59 initial-image strategy selection
540 matched fresh public records, all reproduced exactly; 35 focused tests passed. First geometry batch 47/90 hybrid vs 46/90 v57 and 43/90 v58; second batch 40/90 vs 46/90 and 47/90. Hybrid total 87/180, both comparison policies stronger. Rejected; no default or acceptance change. See `docs/v59_hybrid_routes.md`.

### v60 persistent pose recovery
300 completed final-code public matched records (60 development, 240 fresh). 240/240 fresh records replayed byte-for-byte, 42 focused tests passed. Fresh v57 62/120 vs v60 80/120 (18 improvements, zero regressions), no incorrect map-commit pose recorded by the evaluator in either arm. 134/144 recovery attempts resolved; 151 extra probes. No protected verifier change, default promotion, full regression run, Git commit or acceptance change. See `docs/v60_persistent_pose.md`.

### v62 adaptive calibration

See `docs/v62_adaptive_calibration.md` and `verification/section08/v62/` for the 600 original trial records, 480-record replay comparison, 18 focused tests, and protected-verifier comparison. Candidate remains experimental; 13/31 locally verified sections unchanged.

### v63 online movement-dynamics identification
608 completed original trials including development; 576 fresh/stress records reproduced byte-for-byte. All 288 completed candidate mappings correct across 24 bijections; 247/288 goals vs 23/288 for binary calibration. Two original mapping families 22/24 vs 23/24; comparison is not capacity-controlled. 22 focused tests passed. Protected verifier unchanged. No Git commit, default promotion or full-suite result. See `docs/v63_learned_dynamics.md` and `verification/section08/v63/`.

## v64 context-dependent movement dynamics (experimental)
See `docs/v64_context_dynamics.md` and `verification/section08/v64/`. Matched experimental evidence: v63 with common verified-passage hook 128/256 vs v64 234/256 goals on fresh/stress public scenarios; 111 improved, five regressed. All 242 completed second-stage mappings correct, no incorrect audited pose commitments; 27 focused tests passed; 512/512 fresh/stress records byte-identical replay. The explicit passage trigger, deterministic cardinal bijections, hand-selected public layouts, and unchanged-control regressions limit the result. Full regression not executed, protected verifier unchanged, no candidate promotion or status change.

### v65 observation-triggered dynamics changes (experimental)
See `docs/v65_change_dynamics.md` and `verification/section08/v65/`. 1152 original trial records including repeated development arms; 624 fresh/stress/replan-fresh records exactly reproduced. Strict candidate 70/144 versus v64 86/144 overall, despite improved timed-change results (22/48 vs 12/48). Reidentifications 169/169 correct; no incorrect candidate committed pose observed. 33 focused tests passed; independent holdout and full regression outstanding. No promotion or acceptance change.

### v66 hazard-weighted dynamics (experimental)
See `docs/v66_hazard_dynamics.md` and `verification/section08/v66/`. 624 original records including 480 exactly reproduced fresh/stress and ablation records. 39 focused tests passed. Weighted controller reached 66/96 goals versus 34/96 v65 strict and 47/96 uniform ablation; zero recorded committed pose/mapping errors, 43 correct completed changed-mapping commitments. Fixed prior and deterministic recovery constraints; no full regression, independent holdout, new Git commit or default promotion. All 11 protected verifier files unchanged. No change to section status.
