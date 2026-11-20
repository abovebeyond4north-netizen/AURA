"""Evaluator-only posterior diagnostics; preferred controller unchanged."""
import json, argparse
from pathlib import Path
import numpy as np
from aura.exploration.goal_search import BeliefGoalCamera, PassageRecoveryCamera
from aura.exploration.pose_belief import PoseBelief, SHIFTS
from experiments.goal_acquisition import trial

class InstrumentedCamera(BeliefGoalCamera):
    events = []
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        original=self.pose_belief.update
        def update(before,after,expected=None):
            result=original(before,after,expected)
            self.events.append(dict(expected=expected,shift=result.shift,confidence=result.confidence,probabilities=result.probabilities))
            return result
        self.pose_belief.update=update

def synthetic(seed=85201,n=500):
    rng=np.random.default_rng(seed); rows=[]
    palette=np.array([[0,0,0],[65,65,65],[180,180,180],[255,0,0]],dtype=np.uint8)
    for model in ('independent','burst'):
        for _ in range(n):
            world=palette[rng.choice(4,size=(9,9),p=[.1,.15,.7,.05])]
            shift=SHIFTS[int(rng.integers(len(SHIFTS)))]; dy,dx=shift
            a=world[1:8,1:8].copy(); b=world[1+dy:8+dy,1+dx:8+dx].copy()
            for frame in (a,b):
                frame[3,3]=[0,255,0]
                if model=='independent':
                    mask=rng.random((7,7))<.03
                else:
                    mask=np.zeros((7,7),bool); r,c=rng.integers(0,6,size=2); mask[r:r+2,c:c+2]=True
                frame[mask]=rng.integers(0,256,size=(mask.sum(),3),dtype=np.uint8)
            expected=shift if rng.random()<.8 else SHIFTS[int(rng.integers(len(SHIFTS)))]
            est=PoseBelief().update(a.transpose(2,0,1),b.transpose(2,0,1),expected=expected)
            top=SHIFTS[int(np.argmax(est.probabilities))]
            rows.append(dict(model=model,truth=shift,expected=expected,estimate=est.shift,confidence=est.confidence,correct=top==shift,accepted=est.shift is not None))
    return rows

def run():
    rows=[]
    for seed in range(85100,85105):
        for size,col in ((9,2),(11,4),(13,6)):
            for reverse in (False,True):
                for noise in (0,.03,.08):
                    InstrumentedCamera.events=[]
                    b=trial(seed,size,col,reverse,noise,'frontier',PassageRecoveryCamera)
                    c=trial(seed,size,col,reverse,noise,'frontier',InstrumentedCamera)
                    rows.append(dict(baseline=b,candidate=c,events=InstrumentedCamera.events.copy()))
    return dict(navigation=rows,synthetic=synthetic())
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    with a.output.open('x') as f: json.dump(run(),f,indent=2)
