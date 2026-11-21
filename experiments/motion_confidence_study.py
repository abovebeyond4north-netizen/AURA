"""Paired, non-promoting public changed-dynamics camera regression study.

The confidence learner is advisory. All unsuccessful trials must be retained.
Public developer-owned test seeds are not independent generalization evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from aura.exploration.change_dynamics import EvidenceChangeCamera
from aura.exploration.motion_confidence import ConfidenceRollbackCamera
from aura.exploration.learned_dynamics import PERMUTATIONS
from experiments.change_dynamics_study import trial

CLASSES = (EvidenceChangeCamera, ConfidenceRollbackCamera)


def validate_manifest(m):
    if m.get('protocol') != 'AURA-public-motion-confidence-v1' or m.get('independent') is not False:
        raise ValueError('Manifest must explicitly be a public non-independent study')
    if m.get('controllers') != [x.__name__ for x in CLASSES] or m.get('action_cap') != 65:
        raise ValueError('Cannot silently change study controllers or action cap')
    for key in ('seeds', 'layouts', 'before_permutation_indices', 'observation_noise', 'change_windows'):
        if not isinstance(m.get(key), list) or not m[key]:
            raise ValueError(f'Empty study factor: {key}')
    if len(set(m['seeds'])) != len(m['seeds']) or any(type(x) is not int or x < 0 for x in m['seeds']):
        raise ValueError('Invalid or duplicate seeds')
    if any(type(x) is not int or x < 0 or x >= 24 for x in m['before_permutation_indices']):
        raise ValueError('Invalid control permutations')
    if len(set(m['before_permutation_indices'])) != len(m['before_permutation_indices']):
        raise ValueError('Duplicate mappings')
    if type(m.get('after_offset')) is not int or not 0 < m['after_offset'] < 24:
        raise ValueError('Invalid changed-movement mapping offset')
    if any(not isinstance(l, list) or len(l) != 2 or
           any(type(v) is not int for v in l) or l[0] < 9 or l[0] % 2 != 1 or
           not 1 <= l[1] < l[0] - 1 for l in m['layouts']):
        raise ValueError('Invalid layouts')
    if len({tuple(x) for x in m['layouts']}) != len(m['layouts']):
        raise ValueError('Duplicate layouts')
    if any(type(x) not in (int,float) or not 0 <= x <= .5 for x in m['observation_noise']):
        raise ValueError('Invalid observation corruption')
    if len(set(m['observation_noise'])) != len(m['observation_noise']):
        raise ValueError('Duplicate observation corruption')
    if set(m['change_windows']) != {'none','door','late'} or len(m['change_windows']) != 3:
        raise ValueError('Missing or duplicate change regimes')


def run(m):
    validate_manifest(m)
    rows=[]
    for seed in m['seeds']:
        for size,col in m['layouts']:
            for i in m['before_permutation_indices']:
                for noise in m['observation_noise']:
                    for when in m['change_windows']:
                        before=PERMUTATIONS[i]
                        after=PERMUTATIONS[i if when == 'none' else (i+m['after_offset'])%24]
                        for cls in CLASSES:
                            holder = {}
                            if cls is ConfidenceRollbackCamera:
                                class InstrumentedConfidence(ConfidenceRollbackCamera):
                                    def __init__(self, image, step):
                                        super().__init__(image, step)
                                        holder['camera'] = self
                                used_cls = InstrumentedConfidence
                            else:
                                used_cls = cls
                            row = trial(seed, size, col, before, after, noise, when, used_cls)
                            # Instrumentation is read from the camera itself. The evaluator-only
                            # position remains confined to the standard independent trial audit.
                            row['controller'] = cls.__name__
                            if cls is ConfidenceRollbackCamera:
                                camera = holder.get('camera')
                                if camera is None:
                                    raise ValueError('Experimental camera construction missing')
                                summary = camera.motion_confidence.summary()
                                row['confidence_evidence'] = {
                                    'regime': summary.regime,
                                    'observations': summary.observations,
                                    'abstentions': summary.abstentions,
                                    'contradictions': summary.contradictions,
                                    'rollbacks': summary.rollbacks,
                                    'committed_mapping': summary.committed_mapping,
                                    'support': summary.action_support,
                                    'failed_commits': camera.failed_commits,
                                }
                            else:
                                row['confidence_evidence'] = None
                            if (len(row['actions'])>65 or
                                    any(type(a) is not int or a not in range(5) for a in row['actions'])):
                                raise ValueError('Invalid executed actions')
                            rows.append(row)
    return rows


def summarize(m, rows):
    validate_manifest(m)
    observed=Counter((r['seed'],r['size'],r['column'],tuple(r['before']),
                      tuple(r['after']),r['noise'],r['when'],r['controller']) for r in rows)
    expected=Counter((seed,size,col,PERMUTATIONS[i],
                      PERMUTATIONS[i if when == 'none' else (i+m['after_offset'])%24],
                      noise,when,cls.__name__)
                     for seed in m['seeds'] for size,col in m['layouts']
                     for i in m['before_permutation_indices']
                     for noise in m['observation_noise'] for when in m['change_windows']
                     for cls in CLASSES)
    if observed != expected:
        raise ValueError('Missing, repeated or unexpected paired trials')
    if any(type(r['success']) is not bool or
           not isinstance(r.get('actions'), list) or len(r['actions']) > 65 or
           any(type(a) is not int or a not in range(5) for a in r['actions']) or
           type(r.get('mapping_errors')) is not int or r['mapping_errors'] < 0 or
           not isinstance(r.get('pose_errors'),list) or
           ((r['confidence_evidence'] is None) != (r['controller'] == 'EvidenceChangeCamera'))
           for r in rows):
        raise ValueError('Invalid outcomes, action trace, confidence evidence or audit')
    return {cls.__name__:{'trials':sum(r['controller']==cls.__name__ for r in rows),
                          'successes':sum(r['success'] for r in rows if r['controller']==cls.__name__),
                          'actions_including_failures':sum(len(r['actions']) for r in rows if r['controller']==cls.__name__),
                          'mapping_error_trials':sum(r['mapping_errors']>0 for r in rows if r['controller']==cls.__name__),
                          'pose_error_trials':sum(bool(r['pose_errors']) for r in rows if r['controller']==cls.__name__),
                          'confidence_observations':sum((r['confidence_evidence'] or {}).get('observations',0) for r in rows if r['controller']==cls.__name__),
                          'confidence_abstentions':sum((r['confidence_evidence'] or {}).get('abstentions',0) for r in rows if r['controller']==cls.__name__),
                          'confidence_rollbacks':sum((r['confidence_evidence'] or {}).get('rollbacks',0) for r in rows if r['controller']==cls.__name__),
                          'reasons':dict(Counter(str(r['reason']) for r in rows if r['controller']==cls.__name__ and not r['success']))}
            for cls in CLASSES}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    raw=args.manifest.read_bytes()
    m=json.loads(raw)
    rows=run(m)
    result={'protocol':m['protocol'],'manifest_sha256':hashlib.sha256(raw).hexdigest(),
            'source_commit':os.environ.get('GITHUB_SHA','local-unverified'),
            'summary':summarize(m,rows),'rows':rows,'independent':False,
            'acceptance_promoted':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2)
        f.write('\n')
    print(json.dumps(result['summary'],sort_keys=True))

if __name__=='__main__': main()
