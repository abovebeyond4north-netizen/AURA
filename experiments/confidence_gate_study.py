"""Prespecified matched public confirmation, not a blind/independent evaluation."""
from __future__ import annotations
import argparse
import json
import os
from collections import Counter
from pathlib import Path
from experiments.persistent_pose_study import trial
from aura.exploration.persistent_pose import PersistentPoseCamera
from aura.exploration.composed_routes import ComposedRouteCamera
from aura.exploration.confidence_gated_routes import ConfidenceGatedRouteCamera

CONTROLLERS = (PersistentPoseCamera, ComposedRouteCamera, ConfidenceGatedRouteCamera)
SEEDS = (220051, 220052, 220053, 220054)
LAYOUTS = ((9, 2), (13, 8), (15, 6))
NOISE = (.03, .15, .2)


def run():
    return [trial(seed, size, col, reverse, noise, cls)
            for seed in SEEDS for size, col in LAYOUTS
            for reverse in (False, True) for noise in NOISE
            for cls in CONTROLLERS]


def summarize(rows):
    expected = {(seed, size, col, reverse, noise, cls.__name__)
                for seed in SEEDS for size, col in LAYOUTS
                for reverse in (False, True) for noise in NOISE
                for cls in CONTROLLERS}
    counts = Counter((r['seed'], r['size'], r['column'], r['reverse'],
                      r['noise'], r['controller']) for r in rows)
    if set(counts) != expected or any(n != 1 for n in counts.values()):
        raise ValueError('Missing, duplicate or unexpected paired trials')
    if any(type(r['success']) is not bool or type(r['reverse']) is not bool
           or type(r['pose_errors']) is not list
           or len(r['actions']) > 65
           or any(type(a) is not int or not 0 <= a <= 4 for a in r['actions'])
           or r['pose_errors'] for r in rows):
        raise ValueError('Invalid trace, action budget or pose audit')
    return {cls.__name__: {'successes':sum(r['success'] for r in rows if r['controller']==cls.__name__),
                            'trials':len(SEEDS)*len(LAYOUTS)*2*len(NOISE),
                            'actions':sum(len(r['actions']) for r in rows if r['controller']==cls.__name__)}
            for cls in CONTROLLERS}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    results=run()
    data={'protocol':'AURA-public-confidence-gate-v1','independent':False,'acceptance_promoted':False,
          'source_commit':os.environ.get('GITHUB_SHA','local-unverified'),
          'seeds':SEEDS,'layouts':LAYOUTS,'noise':NOISE,'rows':results,'summary':summarize(results)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as file:
        json.dump(data,file,indent=2)
        file.write('\n')
    print(json.dumps({'summary':data['summary'],'independent':False,'acceptance_promoted':False},sort_keys=True))


if __name__=='__main__':
    main()
