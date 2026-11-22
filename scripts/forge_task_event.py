#!/usr/bin/env python3
"""Opt-in, proposal-only event exporter for Evidence-Gated Skill Forge.

Do not treat an emitted event as independent evaluation or certification.
The caller is responsible for checking that the completed task really succeeded.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import sys

SOURCES = ("agent-cli", "github-actions", "trusted-runner", "manual-export")
FIELDS = ("name", "description", "when", "when_not", "risk", "steps")
RISKS = ("none", "local-write", "external-write", "destructive")
TASK_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")
SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
SENSITIVE = re.compile(
    r"(?i)(-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"\b(?:sk-|gh[pousr]_|github_pat_)[A-Za-z0-9_-]{12,}|"
    r"\b(?:password|access[_-]?token|api[_-]?key|secret)\s*[:=])"
)
MAX_WORKFLOW_FILE = 16_384
MAX_EVENT_LOG = 10_000_000


def read_reviewed_workflow(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > MAX_WORKFLOW_FILE:
        raise ValueError("workflow must be an existing JSON file no larger than 16 KiB")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("workflow must be a JSON object")
    missing = set(FIELDS) - set(value)
    if missing:
        raise ValueError("missing required workflow fields: " + ", ".join(sorted(missing)))
    # Ignore all extra fields; never copy prompts, tool outputs, credentials or metadata.
    clean = {key: value[key] for key in FIELDS}
    slug = clean["name"]
    if not isinstance(slug, str) or len(slug) > 64 or not SLUG.fullmatch(slug):
        raise ValueError("workflow name must be a <=64-character kebab-case slug")
    for name in ("description", "when", "when_not"):
        val = clean[name]
        if not isinstance(val, str) or not val.strip() or len(val) > 1000:
            raise ValueError(f"workflow {name} requires 1-1000 characters")
    steps = clean["steps"]
    if not isinstance(steps, list) or not 2 <= len(steps) <= 30 or any(
        not isinstance(s, str) or not s.strip() or len(s) > 1000 for s in steps
    ):
        raise ValueError("workflow steps must contain 2-30 nonempty strings <=1000 characters")
    if clean["risk"] not in RISKS:
        raise ValueError("workflow risk must use the Forge risk categories")
    text = json.dumps(clean, ensure_ascii=False)
    if SENSITIVE.search(text):
        raise ValueError("workflow appears to contain credential or secret material; review and redact")
    return clean


def append_event(destination: Path, event: dict) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(destination, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(handle, "r+", encoding="utf-8") as out:
        fcntl.flock(out.fileno(), fcntl.LOCK_EX)
        if os.fstat(out.fileno()).st_size > MAX_EVENT_LOG:
            raise ValueError("event log exceeds 10 MB; rotate the file before writing")
        out.seek(0)
        for line in out:
            if not line.strip():
                continue
            prior = json.loads(line)
            if prior.get("source") == event["source"] and prior.get("task_id") == event["task_id"]:
                if prior != event:
                    raise ValueError("event identity already exists with different content")
                return False
        out.seek(0, os.SEEK_END)
        out.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        out.flush()
        os.fsync(out.fileno())
    return True


def emit(workflow: Path, task_id: str, source: str, success: bool,
         exit_code: int, output: Path) -> bool:
    if source not in SOURCES or not TASK_ID.fullmatch(task_id):
        raise ValueError("invalid source or task ID")
    if not success or exit_code != 0:
        raise ValueError("refusing to claim task completion without success=true and exit_code=0")
    if output.resolve() == workflow.resolve():
        raise ValueError("output must not overwrite input workflow")
    candidate = read_reviewed_workflow(workflow)
    return append_event(output, {"event": "task.completed", "source": source,
                                 "task_id": task_id, "success": True, "workflow": candidate})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workflow", type=Path, required=True, help="Human-reviewed workflow JSON")
    p.add_argument("--task-id", required=True, help="Real unique completed-task identifier")
    p.add_argument("--source", choices=SOURCES, required=True)
    p.add_argument("--success", action="store_true", help="Affirm task actually completed successfully")
    p.add_argument("--exit-code", type=int, required=True, help="Observed execution status, zero only")
    p.add_argument("--out", type=Path, default=Path(".skill-forge/events.jsonl"))
    args = p.parse_args(argv)
    try:
        wrote = emit(args.workflow, args.task_id, args.source, args.success, args.exit_code, args.out)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("event recorded (proposal only)" if wrote else "event already recorded (unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
