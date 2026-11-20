"""Run a bounded integrated learning smoke and protected recursive study; no deployment."""
import argparse
import json
from pathlib import Path
from aura.agent import AuraOmega
from aura.config import Config
from aura.self_improvement.study import run_study
from aura.environments.procedural_world import TASKS


def run_learning_matrix(config, output):
    """Independent random-initialization smoke per task; not a transfer study."""
    runs = {}
    for task in TASKS:
        task_config = config.model_copy(update={'log_dir': str(output / task)})
        agent = AuraOmega(task_config)
        try:
            runs[task] = agent.train(task=task, log_every=20)
        finally:
            agent.close()
    return {'tasks': runs, 'scope': 'independent per-task smoke; not mastery or transfer',
            'training_interactions': sum(r['steps'] for r in runs.values()),
            'development_interactions': sum(r['resource_metrics']['development_interactions'] for r in runs.values()),
            'completed_episodes': sum(r['episodes'] for r in runs.values()),
            'all_ledgers_valid': all(r['ledger_valid'] for r in runs.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--all-tasks', action='store_true', help='independent smoke of every procedural family')
    parser.add_argument('--steps-per-task', type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.steps_per_task <= 200:
        parser.error('--steps-per-task must be in [1, 200]')
    args.output.mkdir(parents=True, exist_ok=False)
    config = Config(steps=args.steps_per_task, batch_size=6, warmup=6, latent_dim=8, hidden_dim=16, ensemble_size=2, log_dir=str(args.output / 'learning'), max_episode_steps=20 if args.all_tasks else 40)
    if args.all_tasks:
        learning = run_learning_matrix(config, args.output / 'learning')
    else:
        agent = AuraOmega(config)
        try:
            learning = agent.train(log_every=20)
        finally:
            agent.close()
    recursive = run_study(config, args.output / 'lineage.jsonl', steps=10, replicates=5)
    report = {'learning': learning, 'recursive': recursive, 'scope': 'controlled smoke; not final acceptance'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if recursive['two_accepted_experiments'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
