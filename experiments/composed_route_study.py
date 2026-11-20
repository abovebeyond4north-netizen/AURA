"""Matched evaluation of the composition and both component controls."""
import argparse
import json
from pathlib import Path
from experiments.persistent_pose_study import trial
from aura.exploration.efficient_routes import EfficientRouteCamera
from aura.exploration.persistent_pose import PersistentPoseCamera
from aura.exploration.composed_routes import ComposedRouteCamera


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--layouts',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[trial(a.seed,size,col,rev,noise,cls) for size,col in [tuple(map(int,x.split(':'))) for x in a.layouts.split(',')] for rev in (False,True) for noise in (0,.03,.08) for cls in (EfficientRouteCamera,PersistentPoseCamera,ComposedRouteCamera)]
    with a.output.open('x') as f: json.dump(rows,f,indent=2)


if __name__=='__main__': main()
