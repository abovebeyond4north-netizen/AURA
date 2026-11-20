"""Distill an observed ledger and evaluate closed-loop on public seeds."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from aura.config import seed_everything
from aura.environments.procedural_world import create_environment
from aura.environments.task_generator import task_seed
from aura.memory.evidence_ledger import EvidenceLedger
from aura.planning.distillation import RecurrentImitation,rewarded_episodes,split_episodes,fit_imitation


def rollouts(model, task, cases, *, horizon=80, policy='neural', observation_mode='live'):
    if policy not in ('neural','random') or observation_mode not in ('live','frozen','blank'):
        raise ValueError('Unknown evaluation mode')
    rows=[]
    for label,size,noise,seeds in cases:
        for seed in seeds:
            env=create_environment(task,size=size,max_steps=horizon,noise=noise)
            try:
                obs,_=env.reset(seed=seed); model.reset_episode(); rng=np.random.default_rng(seed+17); total=0.
                start=obs.copy()
                for step in range(1,horizon+1):
                    view=obs if observation_mode=='live' else start if observation_mode=='frozen' else np.zeros_like(obs)
                    action=model.choose(view) if policy=='neural' else int(rng.integers(5))
                    obs,reward,done,trunc,_=env.step(action); total+=reward
                    if done or trunc:break
                rows.append({'case':label,'size':size,'noise':noise,'seed':seed,'success':bool(done),'steps':step,'return':total})
            finally:
                close=getattr(env,'close',None)
                if close is not None:close()
    return rows


def run(ledger_path,output,*,seed=41,updates=120,episodes=10):
    if not 1<=episodes<=100: raise ValueError('Invalid evaluation episode count')
    ledger_path=Path(ledger_path)
    if Path(str(ledger_path)+'-wal').exists() or Path(str(ledger_path)+'-journal').exists():
        raise ValueError('Use a closed, checkpointed evidence ledger')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'source_evidence.sqlite'; shutil.copyfile(ledger_path,snapshot)
    digest=hashlib.sha256(snapshot.read_bytes()).hexdigest()
    ledger=EvidenceLedger(snapshot)
    try:
        demonstrations=rewarded_episodes(ledger)
        all_seeds={r['payload']['seed'] for r in ledger.records('experience')}
    finally: ledger.close()
    tasks={ep[0].context for ep in demonstrations}
    if len(tasks)!=1: raise ValueError('This study requires one task family')
    task=next(iter(tasks)); train,val=split_episodes(demonstrations,seed)
    cases=[]; seen=set(all_seeds)
    for k,(label,size,noise) in enumerate([('same_layout',9,.05),('higher_noise',9,.15),('heavy_noise',9,.5),('larger_layout',11,.05)]):
        seeds=[task_seed('dev',2_000_000_000+seed*10000+k*100+i) for i in range(episodes)]
        if len(set(seeds))!=len(seeds) or seen.intersection(seeds):raise ValueError('Environment seed collision')
        seen.update(seeds)
        cases.append((label,size,noise,seeds))
    seed_everything(seed); model=RecurrentImitation()
    before=rollouts(model,task,cases); random=rollouts(model,task,cases,policy='random')
    fit=fit_imitation(model,train,val,updates=updates)
    after=rollouts(model,task,cases); model.save(output/'imitation.pt')
    frozen=rollouts(model,task,cases,observation_mode='frozen')
    blank=rollouts(model,task,cases,observation_mode='blank')
    restored=rollouts(RecurrentImitation.load(output/'imitation.pt'),task,cases)
    if restored!=after:raise RuntimeError('Reloaded policy differs')
    summary={label:{name:{'successes':sum(r['success'] for r in rows if r['case']==label),'episodes':episodes,
                           'interactions':sum(r['steps'] for r in rows if r['case']==label)}
                    for name,rows in [('untrained',before),('random',random),('trained',after),('frozen_view',frozen),('blank_view',blank)]} for label,*_ in cases}
    if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Evidence snapshot changed')
    report={'scope':'public development only; seed-disjoint, fixed-layout and scale shift; no independent promotion',
            'source_sha256':digest,'source_snapshot_unchanged':hashlib.sha256(snapshot.read_bytes()).hexdigest()==digest,
            'task':task,'seed':seed,'training':fit,'training_episodes':len(train),'validation_episodes':len(val),
            'training_evidence':[[e.evidence_id for e in ep] for ep in train],
            'validation_evidence':[[e.evidence_id for e in ep] for ep in val],
            'training_seeds':sorted({ep[0].seed for ep in train}),'validation_seeds':sorted({ep[0].seed for ep in val}),
            'checkpoint_rollouts_equal':True,'summary':summary,'rows':{'untrained':before,'random':random,'trained':after,'frozen_view':frozen,'blank_view':blank}}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:report[k] for k in ('training_episodes','validation_episodes','training','summary','source_snapshot_unchanged')},indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ledger',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=41);p.add_argument('--updates',type=int,default=120);p.add_argument('--episodes',type=int,default=10)
    a=p.parse_args();run(a.ledger,a.output,seed=a.seed,updates=a.updates,episodes=a.episodes)

if __name__=='__main__':main()
