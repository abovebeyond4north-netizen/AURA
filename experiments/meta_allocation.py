"""Compare meta policies for allocating matched gradient budgets."""
import argparse
import json
from pathlib import Path

import numpy as np
from aura.meta_learning.controller import MetaController
from aura.meta_learning.resource_allocator import Budget, ValueOfComputationAllocator

RATES = {'slow': .001, 'medium': .015, 'fast': .075}
STEPS = 24


def run(seed, policy):
    rng = np.random.default_rng(seed)
    # Random signs and strengths change the prediction problem between runs.
    truth = rng.normal(size=6)
    def samples(n):
        x = rng.normal(size=(n, 6)) * np.array([1, 1, 1, 2, 2, 3])
        return x, x @ truth + rng.normal(0, .1, size=n)

    x,y = samples(128)
    dx,dy = samples(96)
    tx,ty = samples(128)
    weights = np.zeros(6)
    def loss(a,b):
        return float(np.mean((a @ weights-b)**2))
    baseline = loss(tx,ty)
    meta = MetaController(RATES)
    allocation = ValueOfComputationAllocator(RATES, seed=seed, bonus=.02)
    budget = Budget(updates=STEPS)
    trace = []
    for step in range(STEPS):
        if policy == 'meta':
            name = meta.select()
        elif policy == 'value':
            name = allocation.select()
            if name is None:
                break
        elif policy == 'fixed':
            name = tuple(RATES)[step%len(RATES)]
        else:
            name = str(rng.choice(list(RATES)))
        before = loss(dx,dy)
        budget.spend('updates')
        weights -= RATES[name] * (x.T @ (x @ weights-y)) / len(x)
        after = loss(dx,dy)
        # Relative held-out improvement is commensurate across operator choices.
        gain = (before-after)/max(before,1e-12)
        meta.update(name,gain,.01)
        allocation.update(name,gain,.01)
        trace.append(dict(operator=name, before=before,after=after,normalized_gain=gain))
    # This split is never passed to either scheduler.
    return dict(seed=seed,policy=policy,initial=baseline,final=loss(tx,ty),
                updates=budget.used['updates'],trace=trace)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    policies=('meta','value','fixed','random')
    rows=[run(seed,policy) for seed in range(2801,2833) for policy in policies]
    summary={}
    for policy in policies:
        selected=[r for r in rows if r['policy']==policy]
        summary[policy]={'mean_test_mse':float(np.mean([r['final'] for r in selected])),
                         'updates':sum(r['updates'] for r in selected)}
        if policy in ('meta','value'):
            differences = np.array([r['final'] for r in rows if r['policy']=='fixed']) - np.array([r['final'] for r in selected])
            idx=np.random.default_rng(7651).integers(0,len(differences),(10000,len(differences)))
            summary[policy]['paired_fixed_minus_policy_95_percentile']=np.quantile(differences[idx].mean(1),[.025,.975]).tolist()
            summary[policy]['wins_vs_fixed']=int(np.sum(differences>0))
    with a.output.open('x') as f:
        json.dump(dict(summary=summary,runs=rows),f)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
