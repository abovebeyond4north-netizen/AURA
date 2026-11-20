import argparse,json
from itertools import product
from pathlib import Path
from aura.exploration.adaptive_hazard import AdaptiveHazardCamera
from aura.exploration.hazard_dynamics import HazardDynamicsCamera
from aura.exploration.learned_dynamics import PERMUTATIONS
from experiments.change_dynamics_study import trial

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    rows=[]
    for (size,col),i,noise,when,cls in product([(9,4),(11,6)], [1,8,17,22], [0,.08], ['none','door','late'], [HazardDynamicsCamera,AdaptiveHazardCamera]):
        rows.append(trial(a.seed,size,col,PERMUTATIONS[i],PERMUTATIONS[i if when=='none' else (i+11)%24],noise,when,cls))
    with a.output.open('x') as f: json.dump(rows,f,indent=2)
if __name__=='__main__': main()
