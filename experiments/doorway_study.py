"""Public, teacher-assisted layout transfer diagnostic; never a promotion gate."""
import argparse
import json
from pathlib import Path
from aura.config import seed_everything
from aura.environments.procedural_world import ProceduralWorld
from aura.types import Experience
from aura.planning.distillation import RecurrentImitation, fit_imitation

TRAIN=(2,4,6)
RESERVED=(1,3,5,7)

class DoorwayWorld(ProceduralWorld):
    # Experiment-local subclass; production and verifier environments unchanged.
    def __init__(self, column, **kwargs):
        if type(column) is not int or not 1<=column<=7:
            raise ValueError('Invalid doorway column')
        self.column=column
        super().__init__('key', size=9, max_steps=60, **kwargs)

    def reset(self, *, seed=None, options=None):
        # Restore RNG after base reset so only the final observation consumes noise.
        import copy
        import numpy as np
        if seed is not None: self._rng=np.random.default_rng(seed)
        state=copy.deepcopy(self._rng.bit_generator.state)
        super().reset(seed=None, options=options)
        self._rng.bit_generator.state=state
        self.walls.add(self.door)
        self.door=(4,self.column)
        self.walls.discard(self.door)
        return self._observe(), {'task_family':'key'}


def demonstration(column, seed):
    env=DoorwayWorld(column,noise=.05)
    obs,_=env.reset(seed=seed)
    # Explicit privileged teacher, equal shortest path lengths for all columns.
    actions=[1,1]+[3]*(column-1)+[4]+[1]*4+[3]*(7-column)
    result=[]
    for i,action in enumerate(actions):
        nxt,reward,done,trunc,_=env.step(action)
        result.append(Experience(obs,action,nxt,reward,done,trunc,str(seed),f'{seed}:{i}',seed=seed))
        obs=nxt
        if done or trunc: break
    if len(result)!=13 or not result[-1].terminated: raise RuntimeError('Teacher failed')
    return result


def evaluate(model,column,seed):
    env=DoorwayWorld(column,noise=.05); obs,_=env.reset(seed=seed); model.reset_episode()
    for step in range(1,61):
        obs,_,done,trunc,_=env.step(model.choose(obs))
        if done or trunc: break
    return {'column':column,'seed':seed,'success':bool(done),'steps':step}


def run(output, updates=120):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for seed in (41,42,43,44,45):
        for arm in ('fixed','varied'):
            cols=[4]*12 if arm=='fixed' else list(TRAIN)*4
            train=[demonstration(c,seed*1000+i) for i,c in enumerate(cols)]
            val=[demonstration(c,seed*1000+100+i) for i,c in enumerate(cols[:3])]
            seed_everything(seed); model=RecurrentImitation()
            fit=fit_imitation(model,train,val,updates=updates)
            cases=[(c,900000+c*100+i) for c in range(1,8) for i in range(10)]
            results=[evaluate(model,c,s) for c,s in cases]
            path=output/f'{seed}-{arm}.pt'; model.save(path)
            restored=RecurrentImitation.load(path)
            if results!=[evaluate(restored,c,s) for c,s in cases]:
                raise RuntimeError('Checkpoint replay mismatch')
            rows.append({'seed':seed,'arm':arm,'train_columns':cols,'validation_columns':cols[:3],
                         'acquisition_steps':sum(map(len,train))+sum(map(len,val)), 'fit':fit,'results':results})
    report={'scope':'teacher-assisted public diagnostic; no autonomous acquisition or promotion','reserved':RESERVED,'rows':rows}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True); p.add_argument('--updates',type=int,default=120)
    a=p.parse_args(); run(a.output,a.updates)
