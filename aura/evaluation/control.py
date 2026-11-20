"""Bounded development control evaluation; never an independent promotion gate."""
from __future__ import annotations
import numpy as np
from aura.environments.procedural_world import create_environment, TASKS
from aura.environments.task_generator import task_seed


def evaluate_control(world, actor, *, tasks=TASKS, episodes=5, max_steps=40, seed_index=0):
    """Greedy actor versus uniform random, identical episode seeds/horizon.

    The actor sees images only. Evaluation never trains, updates replay or uses
    diagnostics. This public deterministic seed partition is not a secret test.
    Early termination may reduce actual interactions; both caps are identical.
    """
    tasks=tuple(tasks)
    if (not tasks or len(set(tasks))!=len(tasks) or any(t not in TASKS for t in tasks)
        or type(episodes) is not int or not 1<=episodes<=100
        or type(max_steps) is not int or not 1<=max_steps<=1000
        or type(seed_index) is not int or seed_index<0):
        raise ValueError('Invalid development control evaluation contract')
    rows=[]
    for task in tasks:
        for index in range(episodes):
            seed=task_seed('dev',1_000_000_000+seed_index*1000+TASKS.index(task)*100+index)
            for policy in ('actor','random'):
                rng=np.random.default_rng(seed+17)
                env=create_environment(task,max_steps=max_steps,noise=.5 if task=='noise' else .05)
                try:
                    obs,_=env.reset(seed=seed); total=0.
                    for step in range(1,max_steps+1):
                        action=actor.select(world,obs,0.,rng) if policy=='actor' else int(rng.integers(5))
                        obs,reward,done,truncated,_=env.step(action)
                        total+=reward
                        if done or truncated:break
                    rows.append({'task':task,'seed':seed,'policy':policy,'interactions':step,
                                 'return':float(total),'success':bool(done)})
                finally:
                    close=getattr(env,'close',None)
                    if close is not None:close()
    summary={}
    for task in tasks:
        summary[task]={}
        for policy in ('actor','random'):
            selected=[r for r in rows if r['task']==task and r['policy']==policy]
            summary[task][policy]={'successes':sum(r['success'] for r in selected),
                                   'episodes':len(selected),'mean_return':float(np.mean([r['return'] for r in selected])),
                                   'interactions':sum(r['interactions'] for r in selected)}
    return {'scope':'public development control diagnostic; not protected evidence',
            'episode_horizon':max_steps,'episodes_per_task_policy':episodes,'seed_index':seed_index,
            'summary':summary,'rows':rows}
