# Evidence-Gated Skill Forge task-event integration

AURA can opt in to emitting **proposal-only** completed-task events using
`scripts/forge_task_event.py`. The exporter uses the contract supported by the
installed Evidence-Gated Skill Forge `agent_bridge.py`. It is not an evaluator,
a scheduler, an automatic installer, or proof that a new skill is useful.

## 1. Author a reviewed workflow

Write a private JSON file, for example `/tmp/review-workflow.json`, with the
following six fields; never include prompts, credentials, personal data, full
logs, source files, or hidden test contents:

```json
{
  "name": "review-regression-evidence",
  "description": "Review completed regression evidence and summarize pass/fail status.",
  "when": "A new task has an actual test report and a request for evidence review.",
  "when_not": "No task result is available or results are unrelated to regression tests.",
  "risk": "none",
  "steps": [
    "Inspect the actual test report and identify the executed version.",
    "Separate observed passes, failures, and untested claims.",
    "Report the evidence and unresolved blockers without inventing results."
  ]
}
```

This is **only a possible workflow**. Do not represent these examples as three
previously successful real agent tasks. Review the full JSON manually because
pattern-based secret detection is necessarily incomplete.

## 2. Export a real completed task

Only after a real completed task has been independently checked as successful,
record an event with its unique real task ID and observed zero exit code:

```bash
python scripts/forge_task_event.py \
  --workflow /tmp/review-workflow.json \
  --source manual-export \
  --task-id "actual-completed-task-123" \
  --success --exit-code 0 \
  --out .skill-forge/events.jsonl
```

The script whitelists workflow fields, rejects descriptions incompatible with
Forge's 1024-character combined-trigger limit, refuses known prohibited
instruction patterns, rejects malformed/incomplete existing JSONL logs, and
prevents writes through symlinked output files. It refuses failed runs, detects
repeated IDs, restricts event log permissions to mode 0600, and uses an exclusive
file lock for idempotent writes. The pattern-based detection is not a replacement
for manual privacy and security review. It deliberately
does not execute tasks or determine their success: **the caller is responsible
for verifying every completion claim**.

Event storage under `.skill-forge/` is ignored by Git. Keep it private and
rotate it as needed. This tool uses `fcntl` for POSIX systems (Linux/macOS).

## 3. Bridge to the Skill Forge

The Forge scripts live in a **separately installed** Skill Forge package, not
in AURA. Substitute the real skill root in `$FORGE_ROOT`:

```bash
python "$FORGE_ROOT/scripts/agent_bridge.py" \
  --events .skill-forge/events.jsonl \
  --observations .skill-forge/observations.jsonl
python "$FORGE_ROOT/scripts/skill_forge.py" observe \
  --observations .skill-forge/observations.jsonl \
  --inbox .skill-forge/inbox
```

Three distinct successful tasks using an identical reviewed workflow can
produce a *candidate proposal only*. Do not manufacture task IDs, treat CI
reruns as independent real agent outcomes, or copy proposal files into a live
host skill folder without review.

### Proposal-only orchestration

After real task events exist, the AURA wrapper can invoke the two trusted
Forge commands together, under a local lock:

```bash
python scripts/forge_discover.py \
  --forge-root "$FORGE_ROOT" \
  --events .skill-forge/events.jsonl \
  --workspace .skill-forge
```

This runs **only** Forge's `agent_bridge.py` and `skill_forge.py observe`,
not evaluation, attestation, auto-promotion, installation, or any candidate
skill. The environment running this command must already have the trusted Forge
package on disk. The `--forge-root` parameter is an explicit trust decision:
do not point it at an unreviewed third-party directory.

The wrapper reports event ingestion counts and candidate proposals. The
reported task successes come from the event producer and are **not independent
validation**. In particular, repeated GitHub CI reruns are not three distinct
agent task successes. Proposed candidate files remain in
`.skill-forge/inbox` without evidence or auto-installation.

### Verification of adapter and orchestrator

```bash
python -m unittest discover -s tests -p 'test_forge_*.py' -v
```

The orchestrator tests mock the installed Forge subprocesses to verify
argument boundaries and fail-closed behavior. They **do not** establish
end-to-end execution with a real installed Forge or independently verified
skill performance. Perform a separately labeled sandbox smoke test with the
real installed Forge before deployment.

## 4. Evaluation and promotion are separate

A different, access-controlled evaluator must freeze holdouts and run matched
baseline-versus-skill tests. The Forge requires at least eight paired cases,
two contexts, two negative-trigger controls, at least 90% candidate success,
at least 10 percentage points absolute improvement, no baseline-pass
regressions, and paired sign-test p <= 0.05. High-risk workflows require manual
review. Signing requires independently controlled keys and actual agent-runner
evidence; fabricated or local-program tests do not authorize auto-promotion.

The installed Forge's registry promotions do not automatically install skills
in ChatGPT. Any host installation requires a separate explicit operation.

## Test the adapter

```bash
python -m unittest discover -s tests -p 'test_forge_task_event.py' -v
```
