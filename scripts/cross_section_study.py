"""Reproduce v38 live adaptive integration and opt-in reward-route experiment."""
import argparse,json
from pathlib import Path
from aura.agent import AuraOmega
from aura.config import Config
from aura.evaluation.control import evaluate_control


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    common=dict(batch_size=8,warmup=8,latent_dim=8,hidden_dim=16,ensemble_size=2)
    adaptive=Config(seed=42,steps=400,max_episode_steps=40,curiosity_policy='visual_frontier',log_dir=str(output/'adaptive'),**common)
    agent=AuraOmega(adaptive)
    try:
        before=evaluate_control(agent.world_model,agent.actor,episodes=5,max_steps=40,seed_index=38)
        training=agent.train(condition='omega',log_every=40)
        after=evaluate_control(agent.world_model,agent.actor,episodes=5,max_steps=40,seed_index=38)
        adaptive_report={'config':adaptive.model_dump(),'training':training,'before':before,'after':after,
                         'task_steps':{t:sum(r['task']==t for r in agent.history) for t in agent.curriculum.tasks},
                         'curriculum':agent.curriculum.choices}
    finally: agent.close()
    route=Config(seed=1082,steps=640,max_episode_steps=80,curiosity_policy='reward_routes',log_dir=str(output/'route'),**common)
    agent=AuraOmega(route)
    try:
        result=agent.train(condition='curiosity',task='key',log_every=80)
        references={r['record_id'] for r in agent.ledger.records('experience')}
        route_report={'config':route.model_dump(),'training':result,
                      'route_references_resolve':all(set(r['evidence_ids'])<=references for r in result['exploration_metrics']['retained_routes'])}
    finally:agent.close()
    report={'adaptive':adaptive_report,'reward_routes':route_report,
            'scope':'two single-seed development runs; not independent promotion; route seed is diagnostic, not confirmatory'}
    (output/'study.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=run(a.output)
    print(json.dumps({k:v['training'] for k,v in report.items() if k in ('adaptive','reward_routes')},indent=2,allow_nan=False))

if __name__=='__main__':main()
