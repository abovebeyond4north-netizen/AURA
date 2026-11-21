"""Fail-closed action accounting including failed episodes and unknown costs."""
import copy
import os
from pathlib import Path

import pytest

from tools.navigation_resource_audit import account, audit_studies

ROOT = Path(__file__).resolve().parents[1]


def sample(success, arm, seed=19, actions=None, probes=1):
    if actions is None:
        actions = [0, 1, 4]
    return {
        'seed': seed, 'size': 9, 'column': 2, 'reverse': False,
        'noise': 0.03, 'controller': arm, 'success': success,
        'actions': actions, 'pose_errors': [], 'probe_actions': probes,
        'recoveries': 0, 'resolutions': 0,
    }


def test_budget_counts_failure_actions_without_double_counting_probes():
    rows = [
        sample(False, 'persistent', actions=[0,1,4,3], probes=2),
        sample(True, 'composed', actions=[0,1,4], probes=1),
        sample(True, 'persistent', seed=20, actions=[1,2,3], probes=0),
        sample(False, 'composed', seed=20, actions=[0,1], probes=0),
    ]
    result = account(rows, ('persistent', 'composed'))
    p = result['controllers']['persistent']
    c = result['controllers']['composed']
    assert (p['environment_actions'], c['environment_actions']) == (7, 5)
    assert (p['failure_trial_actions'], c['failure_trial_actions']) == (4, 2)
    assert p['internal_probe_actions_included'] == 2
    assert result['paired_comparisons']['composed_minus_persistent']['environment_actions'] == -2
    assert result['resources_not_recorded']['model_training_updates'] is None
    assert result['resources_not_recorded']['full_controller_search_cost'] is None
    assert not result['acceptance_promoted']


@pytest.mark.parametrize('tamper', [
    lambda r: r.pop(),
    lambda r: r.append(copy.deepcopy(r[0])),
    lambda r: r[0].update(actions=list(range(66))),
    lambda r: r[0].update(probe_actions=99),
    lambda r: r[0].update(probe_actions=-1),
    lambda r: r[0].update(success=1),
    lambda r: r[0].update(actions=[0,6]),
    lambda r: r[0].pop('pose_errors'),
    lambda r: r[0].update(recoveries='unknown'),
])
def test_corrupted_or_incomplete_costs_rejected(tamper):
    rows = [sample(False, 'a'), sample(True, 'b')]
    tamper(rows)
    with pytest.raises((ValueError, KeyError, TypeError)):
        account(rows, ('a', 'b'))


def test_both_studies_against_raw_replays():
    p = ROOT / 'runs/public_holdout_v1.json'
    alternate = (ROOT / 'runs/confidence_gate_v1.json',
                 ROOT / 'runs/confidence_gate_public_v1.json')
    q = next((x for x in alternate if x.exists()), alternate[0])
    m = ROOT / 'docs/evaluation/public_holdout_v1.json'
    if not all(x.exists() for x in (p,q,m)):
        if os.environ.get('AURA_REQUIRE_PUBLIC_STUDIES') == '1':
            pytest.fail('Required generated public study data missing')
        pytest.skip('Public data generated later in CI before focused audit')
    result = audit_studies(m.read_bytes(), p.read_bytes(), q.read_bytes())
    a = result['studies']['public']['controllers']
    b = result['studies']['gated']['controllers']
    assert a['EfficientRouteCamera']['environment_actions'] == 860
    assert a['PersistentPoseCamera']['environment_actions'] == 1978
    assert a['ComposedRouteCamera']['environment_actions'] == 1204
    assert b['PersistentPoseCamera']['environment_actions'] == 1938
    assert b['ComposedRouteCamera']['environment_actions'] == 1269
    assert b['ConfidenceGatedRouteCamera']['environment_actions'] == 1904
    assert result['studies']['public']['paired_conditions'] == 72
    assert result['studies']['gated']['paired_conditions'] == 72
    assert not result['independent'] and not result['acceptance_promoted']
