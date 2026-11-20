#!/usr/bin/env python3
"""Run a bounded AURA learner or an active gridworld investigation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/cpu_smoke.yaml')
    p.add_argument('--condition', default='omega',
                   choices=['random', 'fixed', 'curiosity', 'adaptive', 'omega', 'investigate'])
    p.add_argument('--task')
    p.add_argument('--output', type=Path, help='New output directory for a causal investigation')
    p.add_argument('--study-seed', type=int, default=42)
    a = p.parse_args()
    if a.condition == 'investigate':
        from aura.cognition.hypotheses import HypothesisEngine
        from aura.environments.procedural_world import TASKS, create_environment
        from aura.memory.evidence_ledger import EvidenceLedger
        if a.output is None or a.task not in TASKS or a.study_seed < 0:
            p.error('investigate requires --output NEW_DIR, a supported --task and a nonnegative --study-seed')
        a.output.mkdir(parents=True, exist_ok=False)
        ledger = EvidenceLedger(a.output / 'evidence.sqlite')
        try:
            result = HypothesisEngine().investigate_door(
                lambda: create_environment(a.task, max_steps=80), ledger,
                study_id='investigation', seed=a.study_seed)
            if not ledger.verify():
                raise RuntimeError('Invalid study ledger')
        finally:
            ledger.close()
        (a.output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    else:
        from aura.agent import AuraOmega
        from aura.config import Config
        if a.output is not None:
            p.error('--output is for the investigate condition; use log_dir in the training config')
        agent = AuraOmega(Config.read(a.config))
        try:
            result = agent.train(task=a.task, condition=a.condition)
        finally:
            agent.close()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
