"""Matched public benchmark: mapping can change once after verified doorway entry."""
import argparse
import json
from pathlib import Path
from aura.environments.procedural_world import ProceduralWorld
from aura.exploration.door_probe import Unmeasurable
from aura.exploration.goal_search import seek_visible_goal
from aura.exploration.learned_dynamics import LearnedDynamicsCamera, PERMUTATIONS
from aura.exploration.context_dynamics import ContextDynamicsCamera
from experiments.goal_acquisition import Finished


def trial(seed, size, col, before, after, noise, cls):
    env = ProceduralWorld('key', size=size, door_column=col, noise=noise,
                          max_steps=65, diagnostics=True)
    first, _ = env.reset(seed=seed)
    origin = env.diagnostic_state()['agent']
    passed = False
    actions, audits, learned = [], [], []
    success = False
    reason = None
    camera = None
    def step(action):
        nonlocal passed, success
        mapping = after if passed else before if env.has_key else tuple(range(4))
        effective = mapping[action] if action < 4 else action
        image, _, success, truncated, _ = env.step(effective)
        actions.append(int(action))
        if env.diagnostic_state()['agent'] == env.door:
            passed = True
        if success or truncated:
            raise Finished
        return image
    class Audited(cls):
        def _observe(self, image):
            actual = tuple(x-y for x,y in zip(env.diagnostic_state()['agent'], origin))
            if actual != self.position:
                audits.append(dict(action=len(actions), predicted=self.position, actual=actual))
            super()._observe(image)
        def after_key_pickup(self):
            super().after_key_pickup()
            learned.append(('key', self.mapping, before))
    try:
        camera = Audited(first, step)
        if camera.measure(2):
            # Both controllers get the same observable verified-passage hook.
            # The door is physically open and may now be traversed.
            camera.map[camera.position] = 'empty'
            camera.landmarks.pop('door', None)
            if isinstance(camera, ContextDynamicsCamera):
                camera.after_verified_passage()
                learned.append(('passage', camera.mapping, after))
            seek_visible_goal(camera, search=True)
    except Unmeasurable as e:
        reason = str(e)
    except Finished:
        pass
    return dict(seed=seed,size=size,column=col,before=before,after=after,noise=noise,
                controller=cls.__name__,success=success,actions=actions,reason=reason,
                passed=passed,pose_errors=audits,learned=learned,
                mapping_errors=sum(m!=truth for _,m,truth in learned),
                context_probes=getattr(camera,'context_probes',0))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--layouts',required=True)
    p.add_argument('--noise',default='0,.08')
    p.add_argument('--indices',default='0,7,16,23')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[trial(a.seed,size,col,PERMUTATIONS[i],PERMUTATIONS[j],noise,cls)
          for size,col in [tuple(map(int,x.split(':'))) for x in a.layouts.split(',')]
          for i in map(int,a.indices.split(',')) for j in (i,(i+11)%24)
          for noise in map(float,a.noise.split(','))
          for cls in (LearnedDynamicsCamera,ContextDynamicsCamera)]
    with a.output.open('x') as f: json.dump(rows,f,indent=2)

if __name__=='__main__': main()
