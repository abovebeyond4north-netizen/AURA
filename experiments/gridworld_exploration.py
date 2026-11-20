"""Actual procedural-gridworld exploration and noisy-TV stress benchmark.

Every policy sees the exact same image observations and action set. Diagnostics
are read by the evaluator only; policy agents receive neither positions nor
hidden mechanisms. Matched seeds and interaction budgets are enforced.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from aura.environments.procedural_world import ProceduralWorld,spaces
from aura.exploration.validated_progress import ValidatedProgressExplorer
from aura.exploration.visual_frontier import VisualFrontierExplorer

POLICIES=('random','prediction_error','validated_progress','visual_frontier')


class NoisyTVWorld(ProceduralWorld):
    """Add a separate sixth action that shows independent visual noise.

    Actions 0..4 retain their ordinary movement/interaction meaning. Action 5
    changes no object, position or hidden state: it only displays a 3x3 image
    patch of independent RGB noise. This avoids confounding noisy-TV attention
    with the *necessary* switch/door interaction (action 4).
    """
    def __init__(self,**kwargs):
        super().__init__(task='key',**kwargs)
        self.action_space=spaces.Discrete(6)
        self._tv_rng=np.random.default_rng(0)

    def reset(self,*,seed=None,options=None):
        obs,info=super().reset(seed=seed,options=options)
        self._tv_rng=np.random.default_rng(0 if seed is None else seed+10_000)
        return obs,info

    def step(self,action):
        if not self.action_space.contains(action):raise ValueError('Invalid action')
        if int(action)!=5:
            return super().step(action)
        self.steps+=1
        done=self.agent==self.goal
        trunc=self.steps>=self.max_steps and not done
        obs=self._observe().copy()
        obs[:,:3,:3]=self._tv_rng.integers(0,256,size=(3,3,3),dtype=np.uint8)
        return obs,float(1.0 if done else -.01),done,trunc,{'success':done}


def _heldout_prediction_errors(agent, seed, task, steps=120):
    """Evaluate all policies on the SAME fresh pixel transitions for this task.

    Test transitions come from a fixed random behavior policy and are not
    used by the online learner, its validation bank, or action selection.
    """
    env=(NoisyTVWorld if task=='noisy_tv' else ProceduralWorld)(
        **({} if task=='noisy_tv' else {'task':task}), max_steps=50,
        noise=.05 if task not in ('noise','noisy_tv') else 0)
    behavior=np.random.default_rng(500_000+seed)
    obs,_=env.reset(seed=1_000_000+seed*10_000)
    losses=[];persist=[]
    for i in range(steps):
        action=int(behavior.integers(env.action_space.n))
        nxt,_,done,trunc,_=env.step(action)
        x=agent.features(obs); y=agent.features(nxt)
        losses.append(float(np.square(agent.prediction(action,x)-y).mean()))
        persist.append(float(np.square(x-y).mean()))
        obs=nxt
        if done or trunc:
            obs,_=env.reset(seed=1_000_000+seed*10_000+i+1)
    return {'prediction_feature_mse':float(np.mean(losses)),
            'persistence_feature_mse':float(np.mean(persist)),
            'heldout_interactions':steps,
            'heldout_seed_namespace':'dev_1000000'}


def run_one(policy,seed,task='key',steps=500,max_episode_steps=50):
    if policy not in POLICIES or task not in ('key','switch','hidden','altered','noise','noisy_tv'):
        raise ValueError('Invalid benchmark condition')
    if steps<=0 or max_episode_steps<=0:raise ValueError('Budgets must be positive')
    if policy == 'visual_frontier':
        agent=VisualFrontierExplorer(seed=seed+1000,n_actions=6 if task=='noisy_tv' else 5)
    else:
        agent=ValidatedProgressExplorer(seed=seed+1000,n_actions=6 if task=='noisy_tv' else 5,
                                        epsilon=0 if policy=='random' else .08)
    env=(NoisyTVWorld if task=='noisy_tv' else ProceduralWorld)(
        **({} if task=='noisy_tv' else {'task':task}),
        max_steps=max_episode_steps,noise=.05 if task not in ('noise','noisy_tv') else 0,
        diagnostics=True)
    episode=0;actions=[];unique_positions=set();global_positions=set();events={'key':0,'door':0,'switch':0,'goal':0}
    per_episode=[];obs,_=env.reset(seed=seed*10_000+episode)
    for step in range(steps):
        # Control API exposes pixels only; no diagnostic_state in decision path.
        action=agent.choose(obs,policy=policy)
        next_obs,reward,done,trunc,_=env.step(action)
        agent.observe(obs,action,next_obs)
        actions.append(action)
        # Evaluation-only information. Never provided to agent/feature model.
        d=env.diagnostic_state()
        unique_positions.add((episode,d['agent']))
        global_positions.add(d['agent'])
        if d['has_key']:events['key']+=1
        if d['door_open']:events['door']+=1
        if d['switch_on']:events['switch']+=1
        if done:events['goal']+=1
        obs=next_obs
        if done or trunc:
            per_episode.append(bool(done))
            episode+=1
            obs,_=env.reset(seed=seed*10_000+episode)
            if policy == 'visual_frontier':
                agent.reset_episode()
    counts=agent.metrics()
    independent_eval=_heldout_prediction_errors(agent,seed,task)
    return {'policy':policy,'seed':seed,'task':task,'interactions':steps,
            'episodes_completed':len(per_episode),'successful_episodes':sum(per_episode),
            'episode_success_rate':float(np.mean(per_episode)) if per_episode else None,
            'unique_episode_positions':len(unique_positions),
            'unique_physical_positions':len(global_positions),
            'interact_fraction_last_half':float(np.mean(np.asarray(actions)[steps//2:]==4)),
            'tv_fraction_last_half':(float(np.mean(np.asarray(actions)[steps//2:]==5))
                                     if task=='noisy_tv' else None),
            'diagnostic_step_counts':events,'agent_metrics':counts,
            'common_heldout_evaluation':independent_eval,
            'scope':'exploration policy receives only images, actions and transitions; diagnostics for evaluator only'}


def evaluate(seeds=5,steps=500,tasks=('key','switch','noisy_tv'),seed_start=0):
    rows=[run_one(policy,seed,task,steps) for task in tasks
          for seed in range(seed_start,seed_start+seeds) for policy in POLICIES]
    summary={}
    for task in tasks:
        summary[task]={}
        for policy in POLICIES:
            r=[row for row in rows if row['task']==task and row['policy']==policy]
            summary[task][policy]={
                'successful_episodes':sum(a['successful_episodes'] for a in r),
                'episodes_completed':sum(a['episodes_completed'] for a in r),
                'mean_unique_episode_positions':float(np.mean([a['unique_episode_positions'] for a in r])),
                'mean_unique_physical_positions':float(np.mean([a['unique_physical_positions'] for a in r])),
                'mean_interact_fraction_last_half':float(np.mean([a['interact_fraction_last_half'] for a in r])),
                'mean_tv_fraction_last_half':(float(np.mean([a['tv_fraction_last_half'] for a in r]))
                                              if task=='noisy_tv' else None),
                'mean_key_observed_steps':float(np.mean([a['diagnostic_step_counts']['key'] for a in r])),
                'common_heldout_feature_mse':float(np.mean([
                    a['common_heldout_evaluation']['prediction_feature_mse'] for a in r])),
                'common_persistence_feature_mse':float(np.mean([
                    a['common_heldout_evaluation']['persistence_feature_mse'] for a in r]))}
    return {'seeds':seeds,'seed_start':seed_start,'steps_per_seed_policy_task':steps,'summary':summary,
            'rows':rows,'limitations':('Short gridworld exploration diagnostic; no guaranteed task completion, '
                'validation examples collected online, no causal identification; '
                'TV is a sixth no-state-change action, preserving interaction action 4. ')}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=5)
    p.add_argument('--seed-start',type=int,default=0)
    p.add_argument('--steps',type=int,default=500)
    p.add_argument('--output',default='runs/gridworld_exploration.json')
    args=p.parse_args()
    if not 1<=args.seeds<=100 or not 10<=args.steps<=100000:raise SystemExit('Invalid budget')
    result=evaluate(args.seeds,args.steps,seed_start=args.seed_start)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    print(json.dumps(result['summary'],indent=2))
    print('Saved',out)

if __name__=='__main__':main()
