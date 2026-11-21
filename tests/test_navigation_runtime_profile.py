"""Validate measured public CPU and wall-time records without fake training costs."""
import copy
import json
from pathlib import Path

import pytest

from tools.navigation_runtime_profile import CONTROLLERS, validate, validate_manifest

ROOT = Path(__file__).resolve().parents[1]


def demo():
    m = {
        'protocol': 'AURA-public-resource-profile-v1',
        'independent': False, 'action_cap': 65,
        'seeds': [10,11], 'layouts': [[9,2]],
        'reverse_after_key': [False,True],
        'observation_noise': [0.03],
        'controllers': list(CONTROLLERS),
    }
    rows = [
        {'seed': seed, 'size': 9, 'column': 2, 'reverse': reverse,
         'noise': 0.03, 'controller': arm,
         'success': arm == 'PersistentPoseCamera', 'actions': [1,2,3],
         'reason': None, 'pose_errors': [], 'probe_actions': 1,
         'recoveries': 0, 'resolutions': 0,
         'full_episode_wall_ns': 20000, 'full_episode_cpu_ns': 19000,
         'execution_order': i}
        for seed in (10,11) for reverse in (False,True)
        for i,arm in enumerate(CONTROLLERS)
    ]
    return m,rows


def test_measured_profile_reconciles_spend_and_success():
    m,rows = demo()
    result = validate(rows,m)
    assert result['paired_conditions'] == 4
    assert result['controllers']['PersistentPoseCamera']['successes'] == 4
    assert result['controllers']['PersistentPoseCamera']['measured_full_episode_wall_seconds'] > 0
    assert result['resources_not_recorded']['training_wall_seconds'] is None
    assert not result['evaluator_independent']


@pytest.mark.parametrize('mutation',[
    lambda m,r: m.update(independent=True),
    lambda m,r: m['seeds'].append(10),
    lambda m,r: r.pop(),
    lambda m,r: r.append(copy.deepcopy(r[0])),
    lambda m,r: r[0].update(full_episode_wall_ns=0),
    lambda m,r: r[0].update(full_episode_cpu_ns=-1),
    lambda m,r: r[0].update(execution_order=5),
])
def test_rejects_corrupt_profile(mutation):
    m,rows = demo()
    mutation(m,rows)
    with pytest.raises((ValueError,TypeError,KeyError)):
        validate(rows,m)


def test_fixed_public_manifest_is_complete():
    m = json.loads((ROOT/'docs/evaluation/public_resource_profile_v1.json').read_text())
    validate_manifest(m)
    assert len(m['seeds'])*len(m['layouts'])*2*len(m['observation_noise'])*len(CONTROLLERS)==216
