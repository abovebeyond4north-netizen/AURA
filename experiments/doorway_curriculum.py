"""Public failure-driven teacher-assisted curriculum; not a promotion gate."""
import argparse
import json
from pathlib import Path
from aura.config import seed_everything
from aura.planning.distillation import RecurrentImitation, fit_imitation
from experiments.doorway_study import TRAIN, RESERVED, demonstration, evaluate


def allocate(probes, adaptive, count=9):
    if set(probes) != set(TRAIN) or any(not p for p in probes.values()):
        raise ValueError('Development probes must cover only training columns')
    # One coverage episode per column, then failure-weighted allocation.
    chosen = list(TRAIN)
    while len(chosen) < count:
        def score(c):
            failures = 1 + sum(not r['success'] for r in probes[c]) if adaptive else 1
            return failures / (1 + chosen.count(c))
        chosen.append(max(TRAIN, key=score))
    return chosen


def run(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for seed in range(51,56):
        for adaptive in (False,True):
            seed_everything(seed); model=RecurrentImitation()
            base=seed*10000
            warm=[demonstration(c,base+i) for i,c in enumerate(TRAIN)]
            val=[demonstration(c,base+100+i) for i,c in enumerate(TRAIN)]
            first=fit_imitation(model,warm,val,updates=30)
            probes={c:[evaluate(model,c,base+1000+c*10+i) for i in range(5)] for c in TRAIN}
            cols=allocate(probes,adaptive)
            train=warm+[demonstration(c,base+10+i) for i,c in enumerate(cols)]
            second=fit_imitation(model,train,val,updates=90)
            cases=[(c,810000+c*100+i) for c in range(1,8) for i in range(10)]
            results=[evaluate(model,c,s) for c,s in cases]
            path=output/f'{seed}-{adaptive}.pt'; model.save(path)
            restored=RecurrentImitation.load(path)
            if results != [evaluate(restored,c,s) for c,s in cases]:
                raise RuntimeError('Replay mismatch')
            rows.append(dict(seed=seed,adaptive=adaptive,columns=cols,probes=probes,first=first,second=second,results=results, acquisition_steps=195,probe_steps=sum(r['steps'] for rs in probes.values() for r in rs)))
    report=dict(scope='public teacher-assisted diagnostic',reserved=RESERVED,rows=rows)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); run(p.parse_args().output)
