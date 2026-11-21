"""Public-only heldout observed-transition proxy for bounded model proposals.

The target is an actually observed next frame supplied offline. This DOES NOT
measure whether executing suggested actions improves real navigation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from aura.environments.procedural_world import ProceduralWorld
from aura.world_model.dynamics import WorldModel
from aura.planning.bounded_model_planner import BoundedModelPlanner


def validate(m):
    if m.get('protocol') != 'AURA-public-model-plan-proxy-v1' or m.get('independent') is not False:
        raise ValueError('Not the approved non-blinded public protocol')
    train, test=m.get('training_seeds'),m.get('evaluation_seeds')
    if (not isinstance(train,list) or not isinstance(test,list) or
        not train or not test or
        any(type(x) is not int or x<0 for x in train+test) or
        len(train)!=len(set(train)) or len(test)!=len(set(test)) or
        set(train)&set(test)):
        raise ValueError('Missing, duplicated or overlapping train/evaluation seeds')
    for key in ('training_layout','evaluation_layout'):
        layout=m.get(key)
        if (not isinstance(layout,list) or len(layout)!=2 or
            any(type(x) is not int for x in layout) or
            layout[0] < 9 or layout[0]%2!=1 or not 1 <= layout[1] < layout[0]-1):
            raise ValueError('Invalid procedural layout')
    for key in ('steps_per_training_seed','steps_per_evaluation_seed','torch_seed',
                'numpy_training_seed','train_updates','batch_size',
                'latent_dim','hidden_dim','ensemble_size','plan_horizon','plan_quota'):
        x=m.get(key)
        if type(x) is not int or x < (0 if key.endswith('seed') else 1):
            raise ValueError(f'Invalid {key}')
    if m['plan_horizon'] != 1 or m['plan_quota'] != 4 or m.get('action_candidates') != [0,1,2,3]:
        raise ValueError('Inverse-action public proxy requires four one-step candidates')
    if type(m.get('learning_rate')) not in (int,float) or not 0 < m['learning_rate'] < .1:
        raise ValueError('Invalid learning rate')


def collect(seeds, size, col, steps):
    rows=[]
    for seed in seeds:
        world=ProceduralWorld('key',size=size,door_column=col,max_steps=60,noise=0)
        obs,_=world.reset(seed=seed)
        rng=np.random.default_rng(seed+100000)
        for t in range(steps):
            action=int(rng.integers(0,4))
            next_obs,_,terminated,truncated,_=world.step(action)
            rows.append((seed,t,obs.copy(),action,next_obs.copy()))
            obs=next_obs
            if terminated or truncated:
                obs,_=world.reset(seed=seed+1000+t)
    return rows


def batch(rows):
    return np.stack([r[2] for r in rows]),np.asarray([r[3] for r in rows],dtype=np.int64),np.stack([r[4] for r in rows])


def image_mse(world,rows):
    x,a,y=batch(rows)
    world.eval()
    with torch.no_grad():
        observations=torch.as_tensor(x,dtype=torch.float32)/255
        targets=torch.as_tensor(y,dtype=torch.float32)/255
        latent=world.encoder(observations)
        acts=torch.as_tensor(a,dtype=torch.long)
        predicted=torch.stack([h(latent,acts)[1] for h in world.heads]).mean(0)
        return float((predicted-targets).square().mean())


def run(m):
    validate(m)
    torch.set_num_threads(1)
    torch.manual_seed(m['torch_seed'])
    training=collect(m['training_seeds'],*m['training_layout'],m['steps_per_training_seed'])
    evaluation=collect(m['evaluation_seeds'],*m['evaluation_layout'],m['steps_per_evaluation_seed'])
    world=WorldModel(latent_dim=m['latent_dim'],hidden_dim=m['hidden_dim'],
                     ensemble_size=m['ensemble_size'],lr=m['learning_rate'])
    before=image_mse(world,evaluation)
    x,actions,y=batch(training)
    rng=np.random.default_rng(m['numpy_training_seed'])
    train_start=time.monotonic()
    for _ in range(m['train_updates']):
        ids=rng.integers(0,len(training),size=m['batch_size'])
        world.train_batch((x[ids],actions[ids],y[ids]))
    training_seconds=time.monotonic()-train_start
    after=image_mse(world,evaluation)
    planner=BoundedModelPlanner(world,max_horizon=m['plan_horizon'],
                                max_evaluations=m['plan_quota'])
    predictions=[]
    eval_start=time.monotonic()
    for seed,step,obs,actual,target in evaluation:
        proposal=planner.propose(obs,target,allowed_actions=tuple(m['action_candidates']))
        if proposal.status!='proposal' or proposal.evaluated_sequences != 4:
            raise ValueError('Planner failed to evaluate all four allowed actions')
        predictions.append({'seed':seed,'step':step,'actual_action':actual,
                            'proposed_action':proposal.first_action,
                            'match':proposal.first_action==actual,
                            'observed_start_sha256':hashlib.sha256(obs.tobytes()).hexdigest(),
                            'observed_next_sha256':hashlib.sha256(target.tobytes()).hexdigest()})
    evaluation_seconds=time.monotonic()-eval_start
    correct=sum(r['match'] for r in predictions)
    fixed=sum(r['actual_action']==0 for r in predictions)
    return {
        'training_rows':len(training),'evaluation_rows':len(evaluation),
        'train_updates':world.train_steps,'heldout_image_mse_before':before,
        'heldout_image_mse_after':after,'action_proxy_correct':correct,
        'action_proxy_total':len(predictions),'fixed_action0_correct':fixed,
        'training_wall_seconds':training_seconds,'planning_wall_seconds':evaluation_seconds,
        'per_trial':predictions,'independent':False,'acceptance_promoted':False,
        'limitations':['Offline inverse action proxy; an already observed next image is supplied as target',
                       'No suggested actions were executed as real environment goals',
                       'One controlled training run and public seeds; no independent replication',
                       'Timing excludes environment data collection and candidate algorithm development']}


def validate_result(m, data):
    """Refuse missing, duplicate, altered or mislabeled trial evidence."""
    validate(m)
    if (data.get('independent') is not False or
            data.get('acceptance_promoted') is not False):
        raise ValueError('Public experiment must remain nonpromoting')
    trials=data.get('per_trial')
    if not isinstance(trials,list):
        raise ValueError('Missing public per-trial evidence')
    expected={(seed,step) for seed in m['evaluation_seeds']
              for step in range(m['steps_per_evaluation_seed'])}
    observed=[(x['seed'],x['step']) for x in trials]
    if len(trials) != len(expected) or set(observed) != expected:
        raise ValueError('Omitted, duplicated or unexpected evaluation trial')
    if (data.get('action_proxy_total') != len(trials) or
            data.get('action_proxy_correct') != sum(x['match'] for x in trials) or
            data.get('fixed_action0_correct') != sum(x['actual_action']==0 for x in trials)):
        raise ValueError('Result summary differs from recorded outcomes')
    for x in trials:
        if (type(x['actual_action']) is not int or x['actual_action'] not in range(4) or
                type(x['proposed_action']) is not int or x['proposed_action'] not in range(4) or
                type(x['match']) is not bool or
                x['match'] != (x['actual_action']==x['proposed_action']) or
                any(not isinstance(x.get(k),str) or len(x[k])!=64 or
                    any(c not in '0123456789abcdef' for c in x[k])
                    for k in ('observed_start_sha256','observed_next_sha256'))):
            raise ValueError('Corrupt trial action, outcome or observed-frame hash')
    for key in ('heldout_image_mse_before','heldout_image_mse_after'):
        if not np.isfinite(data.get(key,float('nan'))):
            raise ValueError('Nonfinite heldout prediction error')
    return True


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    raw=args.manifest.read_bytes()
    data={'protocol':'AURA-public-model-plan-proxy-v1',
          'manifest_sha256':hashlib.sha256(raw).hexdigest(),
          'source_commit':os.environ.get('GITHUB_SHA','local-unverified'),
          **run(json.loads(raw))}
    validate_result(json.loads(raw),data)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in data.items() if k in
                      ('train_updates','heldout_image_mse_before','heldout_image_mse_after',
                       'action_proxy_correct','action_proxy_total','fixed_action0_correct')},sort_keys=True))

if __name__=='__main__':main()
