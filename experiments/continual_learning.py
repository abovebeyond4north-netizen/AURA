"""Sequential-rule retention study with/without experience replay.
Fixed matched first-stage training, equal second-stage updates, fresh dev seeds.
"""
import argparse,json
from pathlib import Path
import numpy as np
from aura.config import Config,seed_everything
from aura.world_model.dynamics import WorldModel
from protected_verifier.evaluator import collect
from aura.environments.task_generator import task_seed

def fit(model, dataset,steps,seed):
    rng=np.random.default_rng(seed)
    for _ in range(steps):
        idx=rng.integers(0,len(dataset[0]),size=16)
        model.train_batch(tuple(x[idx] for x in dataset))

def run(seed=0,steps=40):
    key_train=collect('key',task_seed('train',seed),120)
    altered_train=collect('altered',task_seed('train',500+seed),120)
    key_dev=collect('key',task_seed('dev',seed),60)
    altered_dev=collect('altered',task_seed('dev',500+seed),60)
    outcomes=[]
    for retain in (False,True):
        seed_everything(seed)
        w=WorldModel(12,24,3,lr=.002)
        fit(w,key_train,steps,seed+8)
        before=w.prediction_loss(key_dev)
        mixture=tuple(np.concatenate([a,b]) for a,b in zip(key_train,altered_train))
        fit(w,mixture if retain else altered_train,steps,seed+9)
        after=w.prediction_loss(key_dev);new=w.prediction_loss(altered_dev)
        outcomes.append({'replay':retain,'retention_loss_before':before,
                         'retention_loss_after':after,'retention_regression':after-before,
                         'new_rule_loss':new})
    return {'seed':seed,'training_updates_per_stage':steps,'outcomes':outcomes}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',type=int,default=3)
    parser.add_argument('--steps',type=int,default=40)
    parser.add_argument('--output',default='runs/continual_report.json')
    args=parser.parse_args()
    report=[run(s,args.steps) for s in range(args.seeds)]
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2))
    print('Saved',dest)
    for use_replay in (False,True):
        values=[next(x['retention_regression'] for x in r['outcomes'] if x['replay']==use_replay) for r in report]
        print('replay',use_replay,'mean_retention_regression',round(float(np.mean(values)),6))
if __name__=='__main__': main()
