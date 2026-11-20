"""Matched route evaluation. Seed ranges are local development/public tests."""
import argparse, json
from pathlib import Path
from experiments.goal_acquisition import trial
from aura.exploration.goal_search import PassageRecoveryCamera
from aura.exploration.observable_routes import ObservableRouteCamera


def run(seed_start, seeds, layouts):
    rows=[]
    for seed in range(seed_start, seed_start+seeds):
        for size,col in layouts:
            for reverse in (False,True):
                for noise in (0,.03,.08):
                    for name, cls in [('baseline',PassageRecoveryCamera),('candidate',ObservableRouteCamera)]:
                        row=trial(seed,size,col,reverse,noise,'frontier',cls)
                        row['controller']=name
                        rows.append(row)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--seed-start',type=int,required=True); p.add_argument('--seeds',type=int,default=5); p.add_argument('--layouts',default='9:2,11:4,13:6'); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    layouts=[tuple(map(int,x.split(':'))) for x in a.layouts.split(',')]
    with a.output.open('x') as f: json.dump(run(a.seed_start,a.seeds,layouts),f,indent=2)
