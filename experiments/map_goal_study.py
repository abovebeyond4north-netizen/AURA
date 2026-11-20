import argparse,json
from pathlib import Path
from experiments.goal_acquisition import trial
from aura.exploration.goal_search import PassageRecoveryCamera, MapRecoveryCamera

def run(start):
    return [dict(arm=label, **trial(seed,size,col,rev,noise,'frontier',cls))
            for seed in range(start,start+10) for size,col in ((9,2),(9,5),(11,4),(11,6))
            for rev in (False,True) for noise in (0,.03)
            for label,cls in [('baseline',PassageRecoveryCamera),('map',MapRecoveryCamera)]]
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--seed-start',type=int,required=True); a=p.parse_args()
    with a.output.open('x') as f: json.dump(run(a.seed_start),f)
