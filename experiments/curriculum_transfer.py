"""Compare three schedules on a bounded action discovery and transfer workbench."""
import argparse
import json
from pathlib import Path

import numpy as np
from aura.curriculum.generator import PrerequisiteCurriculum

GRAPH = {'exploration': (), 'prediction': ('exploration',),
         'acquisition': ('exploration',), 'composition': ('prediction','acquisition'),
         'adaptation': ('composition',)}
TASKS = tuple(GRAPH)
BUDGET = 45


class Workbench:
    """Predictably composable choices with hidden, seed-specific action labels."""
    def __init__(self, seed):
        self.answers = np.random.default_rng(seed).integers(5,size=4).tolist()
        self.memory = {}
        self.failures = set()

    def proposal(self, task):
        a, b, c, d = (self.memory.get(i) for i in range(4))
        prefix = {'exploration': (), 'prediction': (a,), 'acquisition': (a,),
                  'composition': (a,b,c), 'adaptation': (a,c,b)}[task]
        if task == 'composition':
            return prefix
        index = {'exploration':0,'prediction':1,'acquisition':2,'adaptation':3}[task]
        tried = [action for t, action in self.failures if t == task]
        action = self.memory.get(index)
        if action is None:
            action = next(choice for choice in range(5) if choice not in tried)
        return prefix+(action,)

    def outcome(self, task, actions, answers=None):
        a,b,c,d = self.answers if answers is None else answers
        required = {'exploration':(a,), 'prediction':(a,b), 'acquisition':(a,c),
                    'composition':(a,b,c), 'adaptation':(a,c,b,d)}
        return actions == required[task]

    def practice(self, task):
        actions = self.proposal(task)
        success = self.outcome(task,actions)
        index = {'exploration':0,'prediction':1,'acquisition':2,'adaptation':3}.get(task)
        if index is not None and None not in actions:
            if success:
                self.memory[index] = actions[-1]
            else:
                self.failures.add((task,actions[-1]))
        return success


def run(seed, condition):
    trainer = PrerequisiteCurriculum(GRAPH,seed=seed)
    rng = np.random.default_rng(seed)
    workbench = Workbench(seed)
    trace = []
    for step in range(BUDGET):
        if condition == 'adaptive':
            task = trainer.choose_goal()
            detail = trainer.choices[-1]
        else:
            task = TASKS[step % len(TASKS)] if condition == 'fixed' else str(rng.choice(TASKS))
            detail = {'task':task, 'reason':condition}
        won = workbench.practice(task)
        trainer.update(task,won)
        trace.append(dict(detail, success=won))
    # Transfer to a separate family of tickets with a fresh permutation of
    # the public action alphabet. The mapping is provided equally to all arms.
    heldout = []
    evaluator = np.random.default_rng(seed+100009)
    for _ in range(20):
        labels = evaluator.permutation(np.arange(5)+10)
        answers = [int(labels[x]) for x in workbench.answers]
        result = {}
        for task in TASKS:
            predicted = tuple(None if x is None else int(labels[x]) for x in workbench.proposal(task))
            result[task] = workbench.outcome(task,predicted,answers)
        heldout.append(result)
    return dict(seed=seed, condition=condition, trace=trace,
                heldout=heldout,
                downstream_wins=sum(row['adaptation'] for row in heldout))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    rows = [run(seed,condition) for seed in range(110,130)
            for condition in ('adaptive','fixed','random')]
    totals = {term:sum(row['downstream_wins'] for row in rows if row['condition']==term)
              for term in ('adaptive','fixed','random')}
    if not totals['adaptive']>max(totals['fixed'],totals['random']):
        raise AssertionError(totals)
    with args.output.open('x') as f:
        json.dump(dict(budget=BUDGET, totals=totals, runs=rows),f)
    print(totals)


if __name__ == '__main__':
    main()
