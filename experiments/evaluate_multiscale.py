#!/usr/bin/env python3
"""Controlled, seed-matched comparison of residual vs absolute temporal heads.

This is a known dynamical training distribution with held-out random sequences;
its results cannot justify unmeasured gridworld/causal transfer claims.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from aura.world_model.temporal_heads import TemporalHeads
from aura.perception.encoder import ObservationEncoder


def trajectory(seed,steps=16):
    rng=np.random.default_rng(seed)
    image=np.zeros((3,7,7),dtype=np.uint8)
    images=[image.copy()];actions=[];term=[]
    for i in range(steps):
        a=int(rng.integers(5))
        actions.append(a)
        image=image.copy()
        image[a%3]=np.clip(image[a%3].astype(np.int32)+20,0,255)
        images.append(image.copy());term.append(i==steps-1)
    return np.stack(images),actions,term


def benchmark(seeds=(0,1,2,3,4),updates=300):
    data=[]
    for seed in seeds:
        torch.manual_seed(seed)
        encoder=ObservationEncoder(8)
        train=[trajectory(seed*10+i) for i in range(3)]
        heldout=[trajectory(seed*10+100+i) for i in range(3)]
        item={'seed':seed,'training_trajectories':len(train),
              'heldout_trajectories':len(heldout)}
        for mode in ('absolute','residual'):
            torch.manual_seed(seed+100)
            model=TemporalHeads(encoder,8,(1,2,4),hidden_dim=24,lr=.003,
                                freeze_encoder=True,mode=mode)
            init=model.evaluate(heldout)
            before_latent=model.evaluate_latent(heldout)
            model.train_trajectories(train,updates=updates)
            item[mode]={'initial_image_mse':init,'trained_image_mse':model.evaluate(heldout),
                        'initial_latent_mse':before_latent,
                        'trained_latent_mse':model.evaluate_latent(heldout)}
        data.append(item)
    summary={}
    for h in (1,2,4):
        summary[f'h{h}']={
            'persistence_mse':float(np.mean([r['residual']['initial_image_mse'][h] for r in data])),
            'residual_trained_mse':float(np.mean([r['residual']['trained_image_mse'][h] for r in data])),
            'absolute_trained_mse':float(np.mean([r['absolute']['trained_image_mse'][h] for r in data])),
            'residual_latent_mse':float(np.mean([r['residual']['trained_latent_mse'][h] for r in data]))}
    return {'experiment':'multi_horizon_synthetic_action_effects',
            'seeds':list(seeds),'training_updates_per_arm':updates,
            'model':'frozen randomly initialized CNN encoder with per-horizon image/latent heads',
            'results':data,'summary':summary,
            'limitation':'fixed simple visual dynamics; not evidence of performance on unseen rule families'}

if __name__=='__main__':
    report=benchmark()
    p=Path('sample_outputs/multiscale_report.json');p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report['summary'],indent=2))
