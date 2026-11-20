"""Public-seed matched benchmark; evaluator-only control/pose truth."""
import argparse,json
from pathlib import Path
from aura.environments.procedural_world import ProceduralWorld
from aura.exploration.door_probe import Unmeasurable
from aura.exploration.goal_search import seek_visible_goal
from aura.exploration.adaptive_calibration import CalibratedComposedCamera
from aura.exploration.learned_dynamics import LearnedDynamicsCamera, PERMUTATIONS
from experiments.goal_acquisition import Finished


def trial(seed,size,col,mapping,noise,cls):
    env=ProceduralWorld('key',size=size,door_column=col,noise=noise,max_steps=65,diagnostics=True)
    first,_=env.reset(seed=seed)
    origin=env.diagnostic_state()['agent']
    actions=[]; audit=[]; completed=[]; success=False; reason=None; camera=None
    def step(action):
        nonlocal success
        effective=mapping[action] if action<4 and env.has_key else action
        img,_,success,truncated,_=env.step(effective)
        actions.append(int(action))
        if success or truncated: raise Finished
        return img
    class Audited(cls):
        def _observe(self,image):
            actual=tuple(x-y for x,y in zip(env.diagnostic_state()['agent'],origin))
            if actual!=self.position: audit.append(dict(action=len(actions),predicted=self.position,actual=actual))
            super()._observe(image)
        def after_key_pickup(self):
            super().after_key_pickup()
            completed.append(tuple(self.mapping) if hasattr(self,'mapping') else tuple(a^int(self.reversed) for a in range(4)))
    try:
        camera=Audited(first,step)
        if camera.measure(2): seek_visible_goal(camera,search=True)
    except Unmeasurable as e: reason=str(e)
    except Finished: pass
    return dict(seed=seed,size=size,column=col,mapping=mapping,noise=noise,controller=cls.__name__,success=success,actions=actions,reason=reason,pose_errors=audit,completed_calibrations=len(completed),mapping_errors=sum(m!=mapping for m in completed),learning_probes=getattr(camera,'learning_probes',0),mapping_support=getattr(camera,'mapping_support',None))


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--layouts',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--noise',default='0,.08');p.add_argument('--mapping-indices',default='0,7,16,23');a=p.parse_args()
    rows=[trial(a.seed,size,col,PERMUTATIONS[i],noise,cls) for size,col in [tuple(map(int,x.split(':'))) for x in a.layouts.split(',')] for i in map(int,a.mapping_indices.split(',')) for noise in map(float,a.noise.split(',')) for cls in (CalibratedComposedCamera,LearnedDynamicsCamera)]
    with a.output.open('x') as f:json.dump(rows,f,indent=2)
if __name__=='__main__':main()
