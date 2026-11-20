"""Matched-interaction development comparison of frontier and reward-route reuse."""
import argparse,json
from pathlib import Path
import numpy as np
from aura.environments.procedural_world import TASKS,create_environment
from aura.exploration.visual_frontier import VisualFrontierExplorer
from aura.exploration.reward_routes import RewardRouteExplorer


def run_one(policy,task,seed,steps=640,horizon=80):
    if policy not in ('frontier','reward_routes') or task not in TASKS or steps<1 or horizon<1:
        raise ValueError('Invalid experiment contract')
    agent=RewardRouteExplorer(seed+1000) if policy=='reward_routes' else VisualFrontierExplorer(seed=seed+1000)
    env=create_environment(task,max_steps=horizon,noise=.5 if task=='noise' else .05)
    episode=0; wins=[]; first_success=None
    try:
        obs,_=env.reset(seed=seed*10000+episode)
        for i in range(steps):
            action=agent.choose(obs)
            nxt,reward,done,trunc,_=env.step(action)
            if policy=='reward_routes':
                agent.observe(obs,action,nxt,reward=reward,terminated=done,truncated=trunc,evidence_id=f'{seed}:{episode}:{i}')
            else:agent.observe(obs,action,nxt)
            obs=nxt
            if done or trunc:
                wins.append(bool(done)); episode+=1
                if done and first_success is None:first_success=i+1
                obs,_=env.reset(seed=seed*10000+episode); agent.reset_episode()
    finally:
        close=getattr(env,'close',None)
        if close is not None:close()
    return {'policy':policy,'task':task,'seed':seed,'interactions':steps,'completed_episodes':len(wins),
            'successes':sum(wins),'first_success_step':first_success,'metrics':agent.metrics()}


def evaluate(seed_start=90,seeds=3,steps=640,horizon=80):
    rows=[run_one(p,t,s,steps,horizon) for t in TASKS for s in range(seed_start,seed_start+seeds) for p in ('frontier','reward_routes')]
    summary={}
    for task in TASKS:
        summary[task]={}
        for p in ('frontier','reward_routes'):
            r=[v for v in rows if v['task']==task and v['policy']==p]
            summary[task][p]={k:sum(v[k] for v in r) for k in ('interactions','completed_episodes','successes')}
            if p=='reward_routes':summary[task][p]['exact_reproductions']=sum(v['metrics']['exact_successful_reproductions'] for v in r)
    return {'summary':summary,'rows':rows,'horizon':horizon,'scope':'exploratory public development; fixed layout; shared environment seed schedule; no independent promotion'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed-start',type=int,default=90);p.add_argument('--seeds',type=int,default=3);p.add_argument('--steps',type=int,default=640);p.add_argument('--horizon',type=int,default=80);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not 1<=a.seeds<=20 or not 1<=a.steps<=10000 or not 1<=a.horizon<=1000: p.error('invalid budgets')
    result=evaluate(a.seed_start,a.seeds,a.steps,a.horizon)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result['summary'],indent=2))

if __name__=='__main__':main()
