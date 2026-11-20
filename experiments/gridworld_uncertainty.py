"""Frozen risk calibration and rule-shift detection on observed transitions."""
import argparse
import json
from pathlib import Path

import numpy as np

from aura.environments.procedural_world import ProceduralWorld
from aura.world_model.uncertainty import TransitionRisk, UncertaintyEstimator

ACTIONS=(1,3)


def get_transition(seed,action,reverse=False,noise=.02):
    world = ProceduralWorld('key',reverse_after_key=reverse)
    world.reset(seed=seed)
    image,*_ = world.step(1)  # identical observed key pickup state across regimes
    world.noise = noise
    outcome,*_ = world.step(action)
    return image,outcome


def run(seed):
    def indices(split,n):
        return range(seed*100000+split*1000,seed*100000+split*1000+n)
    predictions = {
        action:np.stack([get_transition(i,action)[1] for i in indices(1,64)]).mean(0)
        for action in ACTIONS
    }
    calibration = {
        action:[TransitionRisk.residual(predictions[action],get_transition(i,action)[1])
                for i in indices(2,100)] for action in ACTIONS
    }
    risk = TransitionRisk(calibration)
    check = {}
    for label,reverse,noise in [('normal',False,.02),('reversed',True,.02),('noisy',False,.4)]:
        monitor = UncertaintyEstimator(None)
        residuals = []
        for action in ACTIONS:
            last = None
            for i in indices(3,100):
                image,next_image = get_transition(i,action,reverse,noise)
                monitor.record_observed_outcome(image,action,next_image)
                last = image
                residuals.append(risk.check(action,predictions[action],next_image))
        flags = [r['shifted'] for r in residuals]
        check[label] = dict(flag_rate=float(np.mean(flags)),
                            mse=float(np.mean([r['error'] for r in residuals])),
                            conditional_variance={a:monitor.conditional_outcome_variance(last,a) for a in ACTIONS},
                            evaluated=len(flags))
    mixed = UncertaintyEstimator(None)
    clean = UncertaintyEstimator(None)
    for i in indices(4,100):
        image,y = get_transition(i,1,reverse=(i%2==0),noise=0)
        mixed.record_observed_outcome(image,1,y)
        image,y = get_transition(i,1,noise=0)
        clean.record_observed_outcome(image,1,y)
    return dict(seed=seed,bounds=risk.bounds,groups=check,
                hidden_rule_variance=mixed.conditional_outcome_variance(image,1),
                deterministic_variance=clean.conditional_outcome_variance(image,1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=[run(seed) for seed in (91,103,118,129,140,159,178,199)]
    summary={k:float(np.mean([r['groups'][k]['flag_rate'] for r in rows]))
             for k in ('normal','reversed','noisy')}
    assert summary['normal'] <= .15 and min(summary['reversed'],summary['noisy']) >= .85,summary
    assert all(r['hidden_rule_variance']>0 and r['deterministic_variance']==0 for r in rows)
    assert TransitionRisk({1:[.1]}).check(1,np.zeros((3,7,7)),np.zeros((3,7,7)))['shifted'] is None
    with args.output.open('x') as f:
        json.dump(dict(summary=summary,runs=rows),f,indent=2)
    print(summary)


if __name__=='__main__':
    main()
