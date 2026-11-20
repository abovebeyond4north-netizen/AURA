"""Matched public-seed comparison; evaluator-only audit inherited from v65."""
import argparse,json
from pathlib import Path
from experiments.change_dynamics_study import trial
from aura.exploration.learned_dynamics import PERMUTATIONS
from aura.exploration.context_dynamics import ContextDynamicsCamera
from aura.exploration.change_dynamics import ChangeDynamicsCamera, ReplanningChangeCamera
from aura.exploration.hazard_dynamics import HazardDynamicsCamera

class UniformDynamicsCamera(HazardDynamicsCamera):
    hazard=1.0

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); p.add_argument('--layouts',required=True)
    p.add_argument('--noise',default='0,.08'); p.add_argument('--indices',default='0,7,16,23'); p.add_argument('--ablation',action='store_true'); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    classes=(ReplanningChangeCamera,UniformDynamicsCamera) if a.ablation else (ContextDynamicsCamera,ChangeDynamicsCamera,HazardDynamicsCamera)
    rows=[trial(a.seed,size,col,PERMUTATIONS[i],PERMUTATIONS[i if when=='none' else (i+11)%24],noise,when,cls)
          for size,col in [tuple(map(int,x.split(':'))) for x in a.layouts.split(',')]
          for i in map(int,a.indices.split(',')) for noise in map(float,a.noise.split(',')) for when in ('none','door','late')
          for cls in classes]
    with a.output.open('x') as f: json.dump(rows,f,indent=2)
if __name__=='__main__': main()
