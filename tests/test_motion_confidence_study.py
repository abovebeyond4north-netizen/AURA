"""Experiment cardinality, negative evidence, and public protocol validation."""
import copy
import json
from pathlib import Path

import pytest

from experiments.motion_confidence_study import run, summarize, validate_manifest

ROOT=Path(__file__).resolve().parents[1]


def mini_manifest():
    m=json.loads((ROOT/'docs/evaluation/motion_confidence_public_v1.json').read_text())
    m['seeds']=[98211]
    m['layouts']=[[9,5]]
    m['before_permutation_indices']=[0]
    m['observation_noise']=[0.0]
    return m


def test_public_manifest_preserves_all_changed_regimes_and_no_independence():
    m=json.loads((ROOT/'docs/evaluation/motion_confidence_public_v1.json').read_text())
    validate_manifest(m)
    assert m['independent'] is False
    assert len(m['seeds'])*len(m['layouts'])*len(m['before_permutation_indices'])*len(m['observation_noise'])*len(m['change_windows'])*len(m['controllers']) == 96
    for modification in (lambda q:q.update(independent=True),
                         lambda q:q['controllers'].reverse(),
                         lambda q:q['seeds'].append(q['seeds'][0]),
                         lambda q:q.update(action_cap=80),
                         lambda q:q['change_windows'].remove('late')):
        bad=copy.deepcopy(m)
        modification(bad)
        with pytest.raises(ValueError):validate_manifest(bad)


def test_full_public_study_preserves_failure_records_without_promotion():
    m=mini_manifest()
    rows=run(m)
    result=summarize(m,rows)
    assert result['ConfidenceRollbackCamera']['trials']==3
    assert result['EvidenceChangeCamera']['trials']==3
    assert sum(len(x['actions']) for x in rows)==sum(v['actions_including_failures'] for v in result.values())
    assert all(x['success'] is True or x['reason'] is not None for x in rows)
    assert all(x['confidence_evidence'] is not None for x in rows if x['controller']=='ConfidenceRollbackCamera')
    assert all(x['confidence_evidence'] is None for x in rows if x['controller']=='EvidenceChangeCamera')

    for edit in (lambda q:q.pop(),
                 lambda q:q.append(copy.deepcopy(q[0])),
                 lambda q:q[0].update(seed=9981),
                 lambda q:q[0].update(actions=[8]),
                 lambda q:q[0].update(success=1)):
        corrupted=copy.deepcopy(rows)
        edit(corrupted)
        with pytest.raises((ValueError,TypeError,KeyError)):
            summarize(m,corrupted)


def test_generated_study_receipt_replays_all_outcomes():
    import hashlib
    import os
    path = ROOT/'runs/motion_confidence_public_v1.json'
    if not path.exists():
        if os.environ.get('AURA_REQUIRE_MOTION_STUDY') == '1':
            pytest.fail('Required generated motion confidence evidence is missing')
        pytest.skip('Public study executed after the base regression suite')
    data = json.loads(path.read_bytes())
    manifest_raw = (ROOT/'docs/evaluation/motion_confidence_public_v1.json').read_bytes()
    assert data['manifest_sha256'] == hashlib.sha256(manifest_raw).hexdigest()
    assert data['summary'] == summarize(json.loads(manifest_raw),data['rows'])
    assert data['source_commit']
    assert data['independent'] is False and data['acceptance_promoted'] is False
