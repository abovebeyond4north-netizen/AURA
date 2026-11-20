"""Matched goal-only frontier study with unchanged legacy trial and action cap."""
import argparse,json
from itertools import product
from pathlib import Path
from aura.exploration.door_route_recovery import CalibratedDoorRecoveryCamera
from aura.exploration.frontier_approach import FrontierApproachCamera
from aura.exploration.learned_dynamics import PERMUTATIONS
from experiments.door_route_recovery_study import trial


def run(seed, expanded=False):
    geometries = [(9,3),(11,7)] if not expanded else [(9,2),(11,8),(13,5),(13,9)]
    return [trial(seed,size,col,PERMUTATIONS[i],PERMUTATIONS[i],noise,rate,cls,0)
            for (size,col),i,noise,rate,cls in product(geometries,[2,19],[0,.08],[0,.04,.12],
                                                       [CalibratedDoorRecoveryCamera,FrontierApproachCamera])]

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); p.add_argument('--expanded',action='store_true'); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    with a.output.open('x') as f: json.dump(run(a.seed,a.expanded),f,indent=2)
