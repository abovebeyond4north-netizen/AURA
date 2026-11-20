"""Matched comparison of v56 and risk-aware survey under the same cap."""
import argparse
import json
from pathlib import Path
from experiments.goal_acquisition import trial
from aura.exploration.adaptive_routes import AdaptiveRouteCamera
from aura.exploration.observable_routes import ObservableRouteCamera


def run(seed_start, seeds, layouts):
    return [dict(trial(seed, size, col, rev, noise, 'frontier', cls), controller=name)
            for seed in range(seed_start, seed_start+seeds)
            for size,col in layouts for rev in (False,True) for noise in (0,.03,.08)
            for name,cls in [('baseline',ObservableRouteCamera),('candidate',AdaptiveRouteCamera)]]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-start',type=int,required=True)
    p.add_argument('--seeds',type=int,default=5)
    p.add_argument('--layouts',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    layouts=[tuple(map(int,x.split(':'))) for x in a.layouts.split(',')]
    with a.output.open('x') as f: json.dump(run(a.seed_start,a.seeds,layouts),f,indent=2)

if __name__=='__main__': main()
