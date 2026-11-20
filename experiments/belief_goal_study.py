"""Matched comparison of validated passage camera and posterior fallback."""
import argparse
import json
from pathlib import Path
from experiments.goal_acquisition import trial
from aura.exploration.goal_search import PassageRecoveryCamera, BeliefGoalCamera


def run(seed_start=73100, seeds=5):
    rows=[]
    for seed in range(seed_start, seed_start+seeds):
        for size,col in ((9,2),(9,5),(11,4),(11,6)):
            for reverse in (False,True):
                for noise in (0,.03):
                    for name,cls in [('baseline',PassageRecoveryCamera),('candidate',BeliefGoalCamera)]:
                        record=trial(seed,size,col,reverse,noise,'frontier',camera_class=cls)
                        record['controller']=name
                        rows.append(record)
    summary={}
    for noise in (0,.03):
        summary[str(noise)]={name:{'wins':sum(r['success'] for r in rows if r['noise']==noise and r['controller']==name),
                                   'total':sum(r['noise']==noise and r['controller']==name for r in rows)}
                             for name in ('baseline','candidate')}
    return {'summary':summary,'rows':rows}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed-start',type=int,default=73100)
    p.add_argument('--seeds',type=int,default=5)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=run(args.seed_start,args.seeds)
    with args.output.open('x') as f: json.dump(result,f,indent=2)
    print(result['summary'])

if __name__=='__main__': main()
