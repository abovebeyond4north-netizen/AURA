"""Matched-budget goal acquisition in the primary key gridworld."""
import argparse
import json
from pathlib import Path

import numpy as np

from aura.environments.procedural_world import ProceduralWorld
from aura.exploration.door_probe import Unmeasurable
from aura.exploration.goal_search import seek_visible_goal, GoalCamera
from aura.exploration.movement_calibration import CalibratingCamera


class Finished(Exception):
    pass


def trial(seed,size,column,reverse,noise,policy, camera_class=GoalCamera):
    env = ProceduralWorld('key',size=size,door_column=column,reverse_after_key=reverse,
                           noise=noise,max_steps=65)
    first,_ = env.reset(seed=seed)
    actions = []
    success = False
    uncensored = True
    def step(action):
        nonlocal success, uncensored
        image,_,success,truncated,_ = env.step(action)
        actions.append(int(action))
        if success or truncated:
            uncensored = False
            raise Finished
        return image
    reason = None
    try:
        if policy == 'random':
            rng = np.random.default_rng(seed+801)
            while uncensored:
                step(int(rng.integers(5)))
        else:
            camera = camera_class(first,step)
            if camera.measure(2):
                seek_visible_goal(camera, search=policy=='frontier')
    except Unmeasurable as exc:
        reason = str(exc)
    except Finished:
        pass
    return dict(seed=seed,size=size,column=column,reverse=reverse,noise=noise,
                policy=policy,success=success,actions=actions,reason=reason)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[trial(seed,size,col,rev,noise,policy)
          for seed in range(72100,72110)
          for size,col in ((9,2),(9,5),(11,4),(11,6))
          for rev in (False,True)
          for noise in (0,.03)
          for policy in ('frontier','visible_only','random')]
    summary={}
    for noise in (0,.03):
        summary[str(noise)]={}
        for policy in ('frontier','visible_only','random'):
            group=[row for row in rows if row['policy']==policy and row['noise']==noise]
            summary[str(noise)][policy]=dict(wins=sum(r['success'] for r in group),total=len(group))
    with a.output.open('x') as f:
        json.dump(dict(summary=summary,rows=rows),f)
    print(summary)


if __name__=='__main__':
    main()
