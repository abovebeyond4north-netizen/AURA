import argparse,json
from itertools import product
from pathlib import Path
from aura.exploration.frontier_approach import FrontierApproachCamera
from aura.exploration.local_probe import LocalProbeCamera
from aura.exploration.learned_dynamics import PERMUTATIONS
from experiments.door_route_recovery_study import trial

def run(seed, fresh=False):
    geometries=[(13,9),(9,2)] if not fresh else [(13,8),(15,10),(11,6)]
    return [trial(seed,s,c,PERMUTATIONS[i],PERMUTATIONS[i],noise,rate,cls,0) for (s,c),i,noise,rate,cls in product(geometries,[2,19],[0,.08],[0,.04,.12],[FrontierApproachCamera,LocalProbeCamera])]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--fresh',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with a.output.open('x') as f: json.dump(run(a.seed,a.fresh),f,indent=2)
