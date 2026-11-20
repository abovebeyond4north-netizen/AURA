"""Compare saved shadow generations on common fresh public development seeds."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from aura.config import seed_everything
from aura.environments.task_generator import task_seed
from aura.memory.evidence_ledger import EvidenceLedger
from aura.planning.distillation import RecurrentImitation
from experiments.neural_transfer_study import rollouts


def compare(directory, *, episodes=10):
    directory=Path(directory)
    if type(episodes) is not int or not 1<=episodes<=100:
        raise ValueError('Invalid evaluation episode count')
    study=json.loads((directory/'study.json').read_text())
    config=study['config']; reports=study['result']['imitation_candidates']
    if len(reports)<2 or len({r['task'] for r in reports})!=1 or not config['imitation_inherit']:
        raise ValueError('Expected a single task inherited candidate lineage')
    ledger=EvidenceLedger(directory/'evidence.sqlite')
    try:
        if not ledger.verify() or [r['payload'] for r in ledger.records('imitation_candidate')]!=reports:
            raise ValueError('Candidate evidence mismatch')
        seen={r['payload']['seed'] for r in ledger.records('experience')}
    finally: ledger.close()
    seen.update(s for r in reports for s in r['evaluation_seeds'])
    task=reports[0]['task']; horizon=config['max_episode_steps']
    cases=[]
    for i,(label,size,noise) in enumerate([('same_layout',9,.05),('higher_noise',9,.15),('heavy_noise',9,.5),('larger_layout',11,.05)]):
        seeds=[task_seed('dev',5_000_000_000+i*100+j,secret=f'lineage:{config["seed"]}') for j in range(episodes)]
        if len(set(seeds))!=len(seeds) or seen.intersection(seeds):
            raise ValueError('Evaluation seed collision')
        seen.update(seeds); cases.append((label,size,noise,seeds))
    seed_everything(config['seed']); generations=[]; parent=None; used=set(); ancestral_train=set(); ancestral_held=set(); ancestral_ids=set()
    for r in reports:
        if (r['parent_candidate']!=(parent['candidate'] if parent else None) or
                r['parent_checkpoint_sha256']!=(parent['checkpoint_sha256'] if parent else None) or
                r['generation']!=(parent['generation']+1 if parent else 1)):
            raise ValueError('Invalid candidate lineage')
        fresh=set(r.get('fresh_training_seeds',r['training_seeds']))
        rehearsal=set(r.get('rehearsal_seeds',[]))
        replay_ids={tuple(ids) for ids in r.get('rehearsal_evidence',[])}
        fresh_ids={tuple(ids) for ids in r.get('fresh_training_evidence',r['training_evidence'])}
        if (not rehearsal <= ancestral_train or rehearsal & ancestral_held or fresh & rehearsal
                or fresh | rehearsal != set(r['training_seeds'])
                or not replay_ids <= ancestral_ids or replay_ids & fresh_ids
                or fresh_ids | replay_ids != {tuple(ids) for ids in r['training_evidence']}):
            raise ValueError('Invalid ancestral rehearsal provenance')
        groups=fresh | set(r['validation_seeds'])
        ancestral_train.update(fresh); ancestral_held.update(r['validation_seeds']); ancestral_ids.update(fresh_ids)
        if used.intersection(groups): raise ValueError('Reused ancestral seed group')
        used.update(groups)
        name=f"imitation-candidate-{r['candidate']:04d}.pt"
        if r['checkpoint']!=name: raise ValueError('Invalid checkpoint name')
        data=(directory/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=r['checkpoint_sha256']: raise ValueError('Checkpoint digest mismatch')
        model=RecurrentImitation.load(io.BytesIO(data))
        rows=rollouts(model,task,cases,horizon=horizon)
        restored=rollouts(model,task,[('original_shadow',9,.5 if task=='noise' else .05,r['evaluation_seeds'])],horizon=horizon)
        if [{k:row[k] for k in ('seed','steps','success','return')} for row in restored]!=r['after']:
            raise ValueError('Saved checkpoint does not reproduce its recorded rollout')
        summary={label:{'successes':sum(row['success'] for row in rows if row['case']==label),
                        'episodes':episodes,'interactions':sum(row['steps'] for row in rows if row['case']==label)}
                 for label,*_ in cases}
        generations.append({'generation':r['generation'],'summary':summary,'rows':rows,'restored_shadow_rows_equal':True,
                            'reload_interactions':sum(row['steps'] for row in restored)})
        parent=r
    return {'scope':'public post-training diagnostic; no selection or activation; shared acquisition lineage, not independent replications',
            'task':task,'horizon':horizon,'generations':generations,
            'diagnostic_interactions':sum(row['steps'] for g in generations for row in g['rows']),
            'reload_interactions':sum(g['reload_interactions'] for g in generations)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); report=compare(a.directory,episodes=a.episodes)
    with a.output.open('x') as f: json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('task','diagnostic_interactions','reload_interactions')},indent=2))
    for generation in report['generations']: print(generation['generation'],generation['summary'])
