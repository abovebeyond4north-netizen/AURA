"""Fixed fresh public scenario study; not an independent blind evaluation."""
import json,sys
from itertools import product
from pathlib import Path
from experiments.probe_budget_ablation import Budget6, EntropyProbeCamera, trial
from aura.exploration.learned_dynamics import PERMUTATIONS

def run():
    return [trial(seed,size,col,PERMUTATIONS[i],PERMUTATIONS[i],noise,rate,cls,0)
            for seed,(size,col),i,noise,rate,cls in product(
                (104729,104743),((9,4),(11,6),(15,8)),(5,17),(0,.05),(0,.08),(Budget6,EntropyProbeCamera))]
if __name__=='__main__':
    with Path(sys.argv[1]).open('x') as f: json.dump(run(),f,indent=2)
