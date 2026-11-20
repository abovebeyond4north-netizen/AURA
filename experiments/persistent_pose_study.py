"""Paired evaluation with read-only evaluator-side localization audit."""
import argparse,json
from pathlib import Path
from aura.environments.procedural_world import ProceduralWorld
from aura.exploration.door_probe import Unmeasurable
from aura.exploration.goal_search import seek_visible_goal
from aura.exploration.adaptive_routes import AdaptiveRouteCamera
from aura.exploration.persistent_pose import PersistentPoseCamera
from experiments.goal_acquisition import Finished


def trial(seed,size,col,reverse,noise,cls):
    env=ProceduralWorld('key',size=size,door_column=col,reverse_after_key=reverse,noise=noise,max_steps=65,diagnostics=True)
    first,_=env.reset(seed=seed)
    origin=env.diagnostic_state()['agent']
    actions=[]; audit=[]; success=False; camera=None; reason=None
    def step(action):
        nonlocal success
        img,_,success,truncated,_=env.step(action)
        actions.append(int(action))
        if success or truncated: raise Finished
        return img
    class Audited(cls):
        def _observe(self,image):
            truth=env.diagnostic_state()['agent']
            actual=tuple(x-y for x,y in zip(truth,origin))
            if actual!=self.position: audit.append(dict(action=len(actions),predicted=self.position,actual=actual))
            super()._observe(image)
    try:
        camera=Audited(first,step)
        if camera.measure(2): seek_visible_goal(camera,search=True)
    except Unmeasurable as e: reason=str(e)
    except Finished: pass
    return dict(seed=seed,size=size,column=col,reverse=reverse,noise=noise,controller=cls.__name__,success=success,actions=actions,reason=reason,pose_errors=audit,**{k:getattr(camera,k,0) for k in ('recoveries','resolutions','probe_actions','max_support')})


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--layouts',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[trial(a.seed,size,col,rev,noise,cls) for size,col in [tuple(map(int,x.split(':'))) for x in a.layouts.split(',')] for rev in (False,True) for noise in (0,.03,.08) for cls in (AdaptiveRouteCamera,PersistentPoseCamera)]
    with a.output.open('x') as f:json.dump(rows,f,indent=2)

if __name__=='__main__':main()
