"""Fail-closed exploratory stratification of paired public navigation evidence.

No evaluator independence, confidence certification, or acceptance promotion follows.
The seeds are reused across conditions and must not be treated as iid trials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from tools.public_navigation_study import CONTROLLERS, summarize, validate_manifest

PAIR = ('PersistentPoseCamera', 'ComposedRouteCamera')


def audit(manifest_bytes: bytes, result: dict) -> dict:
    manifest = json.loads(manifest_bytes)
    validate_manifest(manifest)
    if result.get('protocol') != manifest['protocol']:
        raise ValueError('Protocol mismatch')
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if result.get('manifest_sha256') != digest:
        raise ValueError('Evaluation manifest hash mismatch')
    rows = result.get('rows')
    if not isinstance(rows, list):
        raise ValueError('Missing raw trial records')
    summary = summarize(rows, manifest)
    if summary != result.get('summary'):
        raise ValueError('Saved summary differs from raw trial evidence')
    if any(type(r['success']) is not bool or type(r['reverse']) is not bool or
           not isinstance(r['pose_errors'], list) for r in rows):
        raise ValueError('Invalid success, movement or localization record')

    def table(subset: list[dict]) -> dict:
        data = {k: {'successes': sum(r['success'] for r in subset if r['controller'] == k),
                    'trials': sum(r['controller'] == k for r in subset),
                    'failure_reasons': dict(sorted(Counter(str(r['reason']) for r in subset
                                                            if r['controller'] == k and not r['success']).items()))}
                for k in CONTROLLERS}
        labels = lambda r: (r['seed'], r['size'], r['column'], r['reverse'], r['noise'])
        a = {labels(r): r for r in subset if r['controller'] == PAIR[0]}
        b = {labels(r): r for r in subset if r['controller'] == PAIR[1]}
        if set(a) != set(b) or not a:
            raise ValueError('Missing paired controller records')
        gains = sum(b[k]['success'] and not a[k]['success'] for k in a)
        losses = sum(a[k]['success'] and not b[k]['success'] for k in a)
        return {'controllers': data, 'paired_composed_vs_persistent':
                {'gains': gains, 'losses': losses, 'net': gains-losses, 'pairs': len(a)}}

    details = {'overall': table(rows),
               'by_seed': {str(s): table([r for r in rows if r['seed'] == s]) for s in manifest['seeds']},
               'by_layout': {f'{s}x{s}@{col}': table([r for r in rows if (r['size'],r['column'])==(s,col)])
                             for s,col in manifest['layouts']},
               'by_noise': {str(x): table([r for r in rows if r['noise'] == x])
                            for x in manifest['observation_noise']},
               'by_movement': {str(x).lower(): table([r for r in rows if r['reverse'] == x])
                               for x in manifest['reverse_after_key']}}
    return {'source_manifest_sha256': digest,
            'public_and_controlled': True, 'independent': False, 'acceptance_promoted': False,
            'statistical_limitation': 'Four repeated seeds across layouts, corruption, and movement. '
                                     'Subgroup comparisons are exploratory, not independent trials.',
            'details': details}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = audit(args.manifest.read_bytes(), json.loads(args.results.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps({'overall': report['details']['overall'],
                      'by_seed_nets': {k: v['paired_composed_vs_persistent']['net']
                                       for k,v in report['details']['by_seed'].items()},
                      'independent': False, 'acceptance_promoted': False}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
