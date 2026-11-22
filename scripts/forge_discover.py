#!/usr/bin/env python3
"""Opt-in proposal-only AURA -> Evidence-Gated Skill Forge pipeline.

The Forge root must contain operator-reviewed, trusted Forge scripts. This
adapter invokes only the Forge bridge and observation subcommands, never
candidate scripts, attest, scan --auto-promote, or install.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
from pathlib import Path
import subprocess
import sys

MAX_EVENTS_BYTES = 10_000_000
SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
TIMEOUT = 30


class PipelineError(ValueError):
    """Invalid inputs or an unacceptable Forge pipeline result."""


def _secure_workspace(workspace: Path) -> Path:
    if workspace.is_symlink():
        raise PipelineError("workspace must not be a symlink")
    workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not workspace.is_dir():
        raise PipelineError("workspace must be a directory")
    os.chmod(workspace, 0o700)
    return workspace.resolve(strict=True)


def _reject_symlink(path: Path, label: str) -> None:
    if path.is_symlink():
        raise PipelineError(f"{label} must not be a symlink")


def _parse_result(proc: subprocess.CompletedProcess[str], label: str):
    try:
        return json.loads(proc.stdout)
    except (ValueError, TypeError) as exc:
        raise PipelineError(f"{label} produced invalid JSON") from exc


def _invoke(args: list[str], label: str):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=TIMEOUT,
                                check=False)
    except subprocess.TimeoutExpired as exc:
        raise PipelineError(f"{label} exceeded {TIMEOUT} seconds") from exc
    if result.returncode:
        raise PipelineError(f"{label} failed with exit code {result.returncode}")
    return _parse_result(result, label)



def _registry_name_collisions(workspace: Path, observations: Path) -> list[str]:
    """Fail closed on exact-name duplicates in the local Forge registry.

    This does not replace semantic review of other installed host skills.
    """
    registry = workspace / "registry"
    _reject_symlink(registry, "skill registry")
    known: set[str] = set()
    for state in ("installed", "staged"):
        folder = registry / state
        _reject_symlink(folder, "skill registry state")
        if folder.is_dir():
            known.update(p.name for p in folder.iterdir() if p.is_dir())
    if not known:
        return []
    observed: set[str] = set()
    with observations.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict) and isinstance(row.get("workflow"), dict):
                name = row["workflow"].get("name")
                if isinstance(name, str):
                    observed.add(name)
    return sorted(known.intersection(observed))


def pipeline(forge_root: Path, workspace: Path, events: Path) -> dict:
    """Run the installed Forge's proposal-only bridge and observe programs.

    Uses a single advisory lock to avoid competing invocations of this adapter;
    this does not lock arbitrary external Forge invocations.
    """
    root = forge_root.resolve(strict=True)
    bridge_script = root / "scripts" / "agent_bridge.py"
    observe_script = root / "scripts" / "skill_forge.py"
    for script in (bridge_script, observe_script):
        if not script.is_file() or script.is_symlink():
            raise PipelineError(f"expected trusted Forge script is unavailable: {script.name}")
    work = _secure_workspace(workspace)
    observations = work / "observations.jsonl"
    inbox = work / "inbox"
    lock_path = work / ".discovery.lock"
    for path, name in ((events, "events"), (observations, "observations"),
                       (inbox, "candidate inbox"), (lock_path, "lock")):
        _reject_symlink(path, name)
    if not events.is_file() or events.stat().st_size > MAX_EVENTS_BYTES:
        raise PipelineError("events must exist and must not exceed 10 MB")
    if events.resolve() == observations.resolve() or events.resolve() == lock_path.resolve():
        raise PipelineError("events cannot be used as pipeline output")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        os.fchmod(lock.fileno(), 0o600)
        for path, name in ((observations, "observations"), (inbox, "candidate inbox")):
            _reject_symlink(path, name)
        bridged = _invoke([sys.executable, str(bridge_script), "--events", str(events),
                           "--observations", str(observations)], "Forge event bridge")
        if not isinstance(bridged, dict) or bridged.get("promotion_authorized") is not False:
            raise PipelineError("Forge bridge did not confirm proposal-only mode")
        collisions = _registry_name_collisions(work, observations)
        if collisions:
            raise PipelineError("existing local skill names require revision review: "
                                + ", ".join(collisions))
        proposed = _invoke([sys.executable, str(observe_script), "observe", "--observations",
                            str(observations), "--inbox", str(inbox)], "Forge observe")
        if not isinstance(proposed, list) or any(
            not isinstance(p, dict)
            or p.get("action") != "candidate-created-awaiting-evidence"
            or not isinstance(p.get("name"), str)
            or len(p["name"]) > 64
            or not SLUG.fullmatch(p["name"])
            for p in proposed
        ):
            raise PipelineError("Forge reported an unexpected non-proposal action")
        return {"mode": "proposal-only", "events_ingested": bridged.get("ingested"),
                "events_ignored": bridged.get("ignored"), "candidates_proposed": len(proposed),
                "proposal_names": sorted(p["name"] for p in proposed),
                "promoted": 0, "installed": 0,
                "note": "Events and proposal eligibility are untrusted; independent evaluation still required."}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forge-root", required=True, type=Path,
                   help="Trusted installed Evidence-Gated Skill Forge package root")
    p.add_argument("--workspace", type=Path, default=Path(".skill-forge"))
    p.add_argument("--events", type=Path, default=Path(".skill-forge/events.jsonl"))
    a = p.parse_args(argv)
    try:
        result = pipeline(a.forge_root, a.workspace, a.events)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
