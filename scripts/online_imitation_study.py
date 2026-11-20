"""Run autonomous acquisition and scheduled shadow candidates (development only)."""
import argparse
import json
from pathlib import Path
from aura.agent import AuraOmega
from aura.config import Config


def run(output, seed=1082, *, inherit=False, rehearsal=0, budgeted=False):
    if budgeted and not rehearsal:
        raise ValueError("Budgeted rehearsal requires rehearsal episodes")
    if rehearsal and not inherit:
        raise ValueError("Rehearsal requires inheritance")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = Config(seed=seed, steps=640, max_episode_steps=80, curiosity_policy='reward_routes',
                    batch_size=8, warmup=8, latent_dim=8, hidden_dim=16, ensemble_size=2,
                    imitation_interval=8, imitation_updates=120, imitation_max_candidates=3 if inherit else 2,
                    imitation_inherit=inherit, imitation_rehearsal_episodes=rehearsal,
                    imitation_budgeted_rehearsal=budgeted,
                    imitation_eval_episodes=8)
    agent = AuraOmega(config, output)
    try:
        result = agent.train(condition='curiosity', task='key', log_every=80)
        report = {'config': config.model_dump(), 'result': result,
                  'active_policy': 'reward_routes', 'protected_promotion': False,
                  'scope': 'single acquisition seed selected for diagnostic success; matched public shadow rollouts, not independent evaluation'}
    finally:
        agent.close()
    (output/'study.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=1082)
    p.add_argument('--inherit', action='store_true', help='train successive shadow checkpoints on fresh episodes')
    p.add_argument('--rehearsal', type=int, default=0, help='maximum ancestral training episodes per successor')
    p.add_argument('--budgeted', action='store_true', help='cap optimizer sequence work at fresh-only capacity')
    a = p.parse_args()
    report = run(a.output, a.seed, inherit=a.inherit, rehearsal=a.rehearsal, budgeted=a.budgeted)
    result = report['result']
    print(json.dumps({'steps': result['steps'], 'episodes': result['episodes'], 'success_rate': result['success_rate'],
                      'resources': result['resource_metrics'], 'ledger_valid': result['ledger_valid'],
                      'candidates': [{k: r[k] for k in ('step','training','before','after','promotion')} for r in result['imitation_candidates']]}, indent=2))
