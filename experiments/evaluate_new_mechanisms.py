"""Reproducible, controlled skills/causal tests, independent of training smoke."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[1]))
import argparse,json
from pathlib import Path
from aura.skills.online_options import run_skill_transfer
from aura.environments.causal_workbench import CausalWorkbench
from aura.cognition.intervention_study import run_causal_study

def wilson(success,n,z=1.959963984540054):
    import math
    if n<=0: return None
    p=success/n;z2=z*z
    mid=(p+z2/(2*n))/(1+z2/n)
    spread=z*math.sqrt(p*(1-p)/n+z2/(4*n*n))/(1+z2/n)
    return [mid-spread,mid+spread]

def evaluate(seeds=5,primitive_trials=80,composition_trials=60,causal_trials=400):
    if not (1<=seeds<=50 and 10<=primitive_trials<=10000 and 10<=composition_trials<=10000
            and 40<=causal_trials<=10000):
        raise ValueError('Invalid benchmark limits')
    skills=[run_skill_transfer(seed=i,train_episodes=primitive_trials,test_episodes=composition_trials) for i in range(seeds)]
    combined={}
    for policy in ('options','random'):
        wins=sum(r['trained_vs_random'][policy]['wins'] for r in skills)
        total=seeds*composition_trials
        combined[policy]={'wins':wins,'trials':total,'rate':wins/total,'wilson95':wilson(wins,total)}
    causal={}
    for mechanism in ('confounded','effective'):
        causal[mechanism]={
            'passive':run_causal_study(CausalWorkbench(mechanism,seed=7),causal_trials,seed=7,randomized=False),
            'randomized':run_causal_study(CausalWorkbench(mechanism,seed=7),causal_trials,seed=7,randomized=True)}
    return {'skills':{'per_seed':skills,'combined':combined},'causality':causal,
            'scope':'controlled two-stage visual transfer and binary randomized intervention; not broad autonomy',
            'data_disclosure':'scores computed from agent actions, observations and rewards; no hidden mechanism values passed to learner'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=5)
    p.add_argument('--output',default='runs/controlled_mechanisms.json')
    args=p.parse_args()
    result=evaluate(seeds=args.seeds)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    print(json.dumps({'output':str(out),'skills':result['skills']['combined'],
          'causal_classifications':{k:v['randomized']['result'] for k,v in result['causality'].items()}},indent=2))
if __name__=='__main__':main()
