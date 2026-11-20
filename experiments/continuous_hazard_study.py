import argparse,json
from itertools import product
from pathlib import Path
from experiments.repeated_hazard_study import trial
from aura.exploration.learned_dynamics import PERMUTATIONS
from aura.exploration.hazard_dynamics import HazardDynamicsCamera
from aura.exploration.continuous_hazard import ContinuousHazardCamera

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    rows=[]
    for (size,col),i,noise,rate,cls in product([(9,3),(11,7)],[2,19],[0,.08],[0,.04,.12],[HazardDynamicsCamera,ContinuousHazardCamera]):
        rows.append(trial(a.seed,size,col,PERMUTATIONS[i],PERMUTATIONS[i],noise,rate,cls))
    with a.output.open('x') as f: json.dump(rows,f,indent=2)
if __name__=='__main__': main()
