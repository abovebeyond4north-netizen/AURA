"""Matched public scale diagnostic with evaluator-only pose audit."""
import argparse,json
from pathlib import Path
from experiments.persistent_pose_study import trial
from aura.exploration.goal_search import RecoveryGoalCamera
from aura.exploration.composed_routes import ComposedRouteCamera
from aura.exploration.persistent_pose import PersistentPoseCamera


def run(start,count):
    return [trial(seed,size,col,rev,noise,cls)
            for seed in range(start,start+count)
            for size,col in ((9,3),(11,5),(13,4),(13,8),(15,6))
            for rev in (False,True) for noise in (0,.03,.08)
            for cls in (RecoveryGoalCamera,PersistentPoseCamera,ComposedRouteCamera)]

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--count',type=int,default=2)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with a.output.open('x') as f: json.dump(run(a.seed,a.count),f)
