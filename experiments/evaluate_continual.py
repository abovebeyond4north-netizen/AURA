#!/usr/bin/env python3
"""Matched two-stage visual prediction experiment measuring forgetting/retention.

The observable context distinguishes two rule families; targets are generated
from actions, not pretrained labels. This deliberately small controlled process
isolates rehearsal effects, not task-general continual intelligence.
"""
from pathlib import Path
import json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from aura.world_model.dynamics import WorldModel
from aura.meta_learning.plasticity import PlasticityMonitor


def make_task(rule,seed,n=128):
    rng=np.random.default_rng(seed)
    x=np.zeros((n,3,7,7),dtype=np.uint8)
    y=x.copy()
    actions=rng.integers(5,size=n)
    x[:,0,0:2,0:2]=255 if rule else 24
    y[:]=x
    # Visual prediction target; action mapping is not available to the model.
    y[:,2,2:5,2:5]=np.where(((actions+rule*2)%5)<2,255,50)[:,None,None]
    return x,actions,y


def run_one(seed=0,initial_updates=80,adapt_updates=80,replay=False):
    train_old=make_task(0,seed+100)
    train_new=make_task(1,seed+100)
    test_old=make_task(0,seed+200)
    test_new=make_task(1,seed+200)
    torch.manual_seed(seed)
    model=WorldModel(8,16,2,.003,'cpu')
    rng=np.random.default_rng(seed)
    start=time.monotonic()
    for _ in range(initial_updates):
        ids=rng.integers(0,128,16)
        model.train_batch(tuple(a[ids] for a in train_old))
    old_before=model.prediction_loss(test_old)
    for _ in range(adapt_updates):
        if replay:
            old_idx=rng.integers(0,128,8)
            new_idx=rng.integers(0,128,8)
            batch=tuple(np.concatenate((a[old_idx],b[new_idx]),axis=0)
                        for a,b in zip(train_old,train_new))
        else:
            ids=rng.integers(0,128,16)
            batch=tuple(a[ids] for a in train_new)
        model.train_batch(batch)
    elapsed=time.monotonic()-start
    plasticity=PlasticityMonitor(.01).measure(model,test_old[0])
    return {'plasticity':plasticity,'seed':seed,'replay':replay,'old_mse_after_stage1':old_before,
            'old_mse_after_stage2':model.prediction_loss(test_old),
            'new_mse_after_stage2':model.prediction_loss(test_new),
            'updates':initial_updates+adapt_updates,
            'rehearsal_buffer_bytes':sum(a.nbytes for a in train_old) if replay else 0,
            'wall_seconds':elapsed}


def experiment(seeds=range(5),initial_updates=80,adapt_updates=80):
    rows=[]
    for seed in seeds:
        rows.append({'seed':seed,
                     'no_replay':run_one(seed,initial_updates,adapt_updates,False),
                     'with_replay':run_one(seed,initial_updates,adapt_updates,True)})
    summary={}
    for mode in ('no_replay','with_replay'):
        summary[mode]={key:float(np.mean([r[mode][key] for r in rows]))
                       for key in ('old_mse_after_stage1','old_mse_after_stage2',
                                   'new_mse_after_stage2','wall_seconds')}
        summary[mode]['mean_forgetting_mse']=summary[mode]['old_mse_after_stage2']-summary[mode]['old_mse_after_stage1']
    return {'experiment':'context_conditioned_visual_rule_forgetting',
            'seeds':list(seeds),'updates_per_phase':[initial_updates,adapt_updates],
            'matched_minibatch_size':16,'same_initial_model_weights_per_seed':True,
            'details':rows,'summary':summary,
            'limitations':['small synthetic contextual rule-switch, not complex procedural gridworld',
                           'replay halves current task data per step; adaptation may slow',
                           '5-seed descriptive evidence, not pre-registered final promotion proof']}

if __name__=='__main__':
    result=experiment()
    target=Path('sample_outputs/continual_rule_shift_report.json')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result['summary'],indent=2))
