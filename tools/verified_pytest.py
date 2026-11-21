"""Run a bounded pytest partition with durable stdout, JUnit and exit evidence."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--label", default="full")
    p.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.timeout <= 0 or "/" in args.label or ".." in args.label:
        p.error("Invalid timeout or label")
    dest = args.output_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    report = dest / (args.label + ".json")
    log = dest / (args.label + ".log")
    junit = dest / (args.label + ".xml")
    extra = args.pytest_args
    if extra[:1] == ["--"]:
        extra = extra[1:]
    cmd = [sys.executable, "-m", "pytest", "-ra", "--junitxml=" + str(junit), *extra]
    start = time.monotonic()
    status = {"label": args.label, "command": cmd, "state": "started", "exit_code": None,
              "timeout": False, "duration_seconds": None, "junit": str(junit), "log": str(log),
              "source_commit": os.environ.get("GITHUB_SHA", "local-unverified"),
              "python": sys.version.split()[0]}
    atomic_json(report, status)
    proc = None
    try:
        with log.open("wb", buffering=0) as out:
            proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=args.timeout)
                status["state"] = "finished"
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                code = 124
                status.update(state="timed_out", timeout=True)
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        status["state"] = "runner_error"
        status["error"] = repr(exc)
        code = 125
    status.update(exit_code=code, duration_seconds=round(time.monotonic() - start, 3))
    atomic_json(report, status)
    print(json.dumps(status, sort_keys=True), flush=True)
    return code

if __name__ == "__main__":
    raise SystemExit(main())
