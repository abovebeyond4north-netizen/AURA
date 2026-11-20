"""Public matched sensitivity study; no simulator-derived hazard supplied per trial."""
import argparse,json
from itertools import product
from pathlib import Path
from aura.exploration.door_route_recovery import CalibratedDoorRecoveryCamera
from aura.exploration.learned_dynamics import PERMUTATIONS
from experiments.door_route_recovery_study import trial
class MediumHazard(CalibratedDoorRecoveryCamera): hazard=.08
class HighHazard(CalibratedDoorRecoveryCamera): hazard=.15

def run(seed):
    return [trial(seed,size,col,PERMUTATIONS[i],PERMUTATIONS[i],noise,rate,cls,0) for (size,col),i,noise,rate,cls in product([(9,3),(11,7)],[2,19],[0,.08],[0,.04,.12],[CalibratedDoorRecoveryCamera,MediumHazard,HighHazard])]
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    with a.output.open('x') as f: json.dump(run(a.seed),f,indent=2)
