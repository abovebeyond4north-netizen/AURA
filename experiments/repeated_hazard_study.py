"""Matched public study with evaluator-only unsignaled rule transitions."""
import argparse,json,random
from pathlib import Path
from aura.environments.procedural_world import ProceduralWorld
from aura.exploration.door_probe import Unmeasurable
from aura.exploration.goal_search import seek_visible_goal
from aura.exploration.learned_dynamics import LearnedDynamicsCamera, PERMUTATIONS
from aura.exploration.context_dynamics import ContextDynamicsCamera
from aura.exploration.change_dynamics import ChangeDynamicsCamera, EvidenceChangeCamera, ReplanningChangeCamera
from experiments.goal_acquisition import Finished


def trial(seed,size,col,before,after,noise,when,cls):
    env=ProceduralWorld('key',size=size,door_column=col,noise=noise,max_steps=65,diagnostics=True)
    first,_=env.reset(seed=seed)
    origin=env.agent
    passed=False; switched=False; key_steps=0; success=False; reason=None; camera=None
    actions=[]; audits=[]; learned=[]; changes=[]
    rng=random.Random(seed*1009+size*31+col); current=before
    def step(action):
        nonlocal passed,switched,key_steps,success,current
        if env.has_key:
            if key_steps>=6 and rng.random()<when:
                current=rng.choice(PERMUTATIONS); changes.append(len(actions)+1)
            key_steps+=1
        mapping=current if env.has_key else tuple(range(4))
        image,_,success,truncated,_=env.step(mapping[action] if action<4 else action)
        actions.append(int(action))
        if env.agent==env.door: passed=True
        if success or truncated: raise Finished
        return image
    class Audited(cls):
        def _observe(self,image):
            actual=tuple(x-y for x,y in zip(env.agent,origin))
            if actual!=self.position: audits.append((len(actions),self.position,actual))
            super()._observe(image)
        def after_key_pickup(self):
            super().after_key_pickup()
            learned.append(('key',self.mapping,current))
        def _commit_reidentified(self,mapping,pos,image):
            super()._commit_reidentified(mapping,pos,image)
            learned.append(('detected',mapping,current))
    try:
        camera=Audited(first,step)
        if camera.measure(2):
            camera.map[camera.position]='empty'; camera.landmarks.pop('door',None)
            if isinstance(camera,ContextDynamicsCamera):
                camera.after_verified_passage()
                learned.append(('passage',camera.mapping,current))
            seek_visible_goal(camera,search=True)
    except Unmeasurable as e: reason=str(e)
    except Finished: pass
    return dict(seed=seed,size=size,column=col,before=before,after=after,noise=noise,when=when,
                controller=cls.__name__,success=success,actions=actions,reason=reason,passed=passed,
                change_steps=changes,pose_errors=audits,learned=learned,
                mapping_errors=sum(m!=truth for _,m,truth in learned),
                estimated_rate=getattr(getattr(camera,'rate',None),'mean',None), change_events=getattr(camera,'change_events',0),change_probes=getattr(camera,'change_probes',0),
                change_resolutions=getattr(camera,'change_resolutions',0),context_probes=getattr(camera,'context_probes',0))



def main():
    from itertools import product
    from aura.exploration.hazard_dynamics import HazardDynamicsCamera
    from aura.exploration.adaptive_hazard import AdaptiveHazardCamera
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    rows=[trial(a.seed,size,col,PERMUTATIONS[i],PERMUTATIONS[i],noise,rate,cls) for (size,col),i,noise,rate,cls in product([(9,3),(11,7)],[2,19],[0,.08],[0,.04,.12],[HazardDynamicsCamera,AdaptiveHazardCamera])]
    with a.output.open('x') as f: json.dump(rows,f,indent=2)
if __name__=='__main__': main()
