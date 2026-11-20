"""Matched interaction budgets; outputs raw descriptive metrics, not significance."""
import argparse,json
from pathlib import Path
from aura.agent import AuraOmega
from aura.config import Config
from aura.evaluation.runner import evaluate_world_model

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/cpu_smoke.yaml');p.add_argument('--seeds',type=int,default=2)
    a=p.parse_args(); config=Config.read(a.config); reports=[]
    for condition in ('random','fixed','curiosity','adaptive','omega'):
        for seed in range(a.seeds):
            conf=config.model_copy(update={'seed':config.seed+seed,'log_dir':str(Path(config.log_dir)/'baselines'/condition/str(seed))})
            agent=AuraOmega(conf)
            metrics=agent.train(condition=condition)
            evals=evaluate_world_model(agent.world_model,'key',seed)
            reports.append({'condition':condition,'seed':conf.seed,**metrics,**evals})
            agent.close()
    dest=Path(config.log_dir)/'baseline_report.json';dest.write_text(json.dumps(reports,indent=2))
    print(json.dumps({'report':str(dest),'runs':len(reports)},indent=2))
if __name__=='__main__': main()
