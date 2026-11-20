"""Matched strategy assessment. Selection uses initial pixels only."""
import argparse,json
from pathlib import Path
from experiments.goal_acquisition import trial
from aura.exploration.adaptive_routes import AdaptiveRouteCamera
from aura.exploration.efficient_routes import EfficientRouteCamera
from aura.exploration.hybrid_routes import HybridRouteCamera


def run(seed, layouts):
    rows=[]
    for size,col in layouts:
        for reverse in (False,True):
            for noise in (0,.03,.08):
                for name,cls in (('v57',AdaptiveRouteCamera),('v58',EfficientRouteCamera),('v59',HybridRouteCamera)):
                    row = trial(seed,size,col,reverse,noise,'frontier',cls)
                    rows.append(dict(row,controller=name))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--layouts',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=run(a.seed,[tuple(map(int,x.split(':'))) for x in a.layouts.split(',')])
    with a.output.open('x') as f: json.dump(rows,f,indent=2)
