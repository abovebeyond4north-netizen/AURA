"""Rebuild missing historical CI fixtures; reject any non-identical replay.

These datasets are *historical development evidence*, not blind evaluation.
Do not silently replace the originals when a hash differs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

FIXTURES = {
    "verification/section08/v76/development.json": ("1af84999d805b6c2f91f0d15d5e1805e9d6906df1c4e5d05ef6318d34bcc80d2", "v76"),
    "verification/section08/passage_dev.json": ("5e6c03ae7789ffe61dc26d44e14692a6a71b6a27751132a208368d12adbccdf7", "passage-dev"),
    "verification/section08/passage_fresh.json": ("4e56973534674170141f12e59944cfab55e0378d73fea9bf65be0c951ebf0b32", "passage-fresh"),
    "verification/section08/recovery_fresh.json": ("5bb2cc2623546e73a58f0f41db7ad9589be3d16d0ce6f65c00723334c43d8103", "recovery-fresh"),
}

def reproduce(kind: str):
    if kind == "v76":
        from experiments.belief_lookahead_study import run
        return json.dumps(run(98205, False), indent=2)
    if kind == "passage-dev":
        from experiments.passage_goal_study import run
        return json.dumps(run(74100))
    if kind == "passage-fresh":
        from experiments.passage_goal_study import run
        return json.dumps(run(75100))
    if kind == "recovery-fresh":
        from experiments.recovery_goal_study import run
        return json.dumps(run(74100))
    raise ValueError(kind)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, default=Path("runs/fixture_replay.json"))
    args = parser.parse_args()
    report = []
    failures = []
    for relative, (expected, kind) in FIXTURES.items():
        path = args.root / relative
        rebuilt = reproduce(kind).encode("utf-8")
        actual = hashlib.sha256(rebuilt).hexdigest()
        existing = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        valid = actual == expected and (existing is None or existing == expected)
        report.append({"path": relative, "expected_sha256": expected, "replayed_sha256": actual,
                       "existing_sha256": existing, "match": valid})
        if not valid:
            failures.append(relative)
            continue
        if existing is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(rebuilt)
    output = args.root / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"replays": report, "all_match": not failures}, indent=2) + "\n")
    for entry in report:
        print(("PASS" if entry["match"] else "FAIL"), entry["path"])
    return int(bool(failures))

if __name__ == "__main__":
    raise SystemExit(main())
