"""Finite-budget comparison where a larger computation batch costs more."""
import argparse
import json
from pathlib import Path

import numpy as np
from aura.meta_learning.resource_allocator import Budget, ValueOfComputationAllocator

PRICE = {'small':1, 'full':8}
LIMIT = 64


def run(seed, mode):
    rng = np.random.default_rng(seed)
    target = rng.normal(size=8)
    def data(count):
        x = rng.normal(size=(count,8))
        return x,x@target+rng.normal(0,.5,size=count)
    train = data(128)
    dev = data(128)
    test = data(256)
    w = np.zeros(8)
    def loss(split):
        x,y = split
        return float(np.mean((x@w-y)**2))
    budget = Budget(updates=LIMIT)
    allocator = ValueOfComputationAllocator(PRICE,bonus=.03)
    trace = []
    while True:
        choices = [name for name,cost in PRICE.items() if budget.can_spend('updates',cost)]
        if not choices:
            break
        if mode == 'adaptive':
            name = allocator.select(choices)
            if name is None:
                break
        elif mode == 'fixed':
            name = 'full' if 'full' in choices else 'small'
        else:
            name = str(rng.choice(choices))
        cost = PRICE[name]
        before = loss(dev)
        x,y = train
        idx = rng.choice(len(x),16*cost,replace=False)
        w -= .20 * (x[idx].T @ (x[idx]@w-y[idx]))/len(idx)
        budget.spend('updates',cost)
        after = loss(dev)
        allocator.update(name,(before-after)/max(before,1e-12),.01*cost)
        trace.append(dict(batch=name, units=cost, development_before=before,development_after=after))
    return dict(seed=seed, policy=mode, final=loss(test), units=budget.used['updates'], trace=trace)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    policies=('adaptive','fixed','random')
    rows=[run(seed,policy) for seed in range(9311,9343) for policy in policies]
    summary={name:dict(mean_test_mse=float(np.mean([row['final'] for row in rows if row['policy']==name])),
                       mean_units=float(np.mean([row['units'] for row in rows if row['policy']==name])))
             for name in policies}
    with args.output.open('x') as f:
        json.dump(dict(limit=LIMIT,summary=summary,runs=rows),f)
    print(summary)


if __name__=='__main__':
    main()
