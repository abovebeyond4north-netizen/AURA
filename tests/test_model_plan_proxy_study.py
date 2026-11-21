"""Validate non-promoting inverse action proxy and reject corrupted trial output."""
import copy
import json
from pathlib import Path

import pytest

from experiments.model_plan_proxy_study import validate, run, validate_result

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    return json.loads((ROOT/'docs/evaluation/model_plan_proxy_v1.json').read_text())


def test_manifest_separates_evaluation_and_training_seeds():
    m=fixture()
    validate(m)
    for mutation in (lambda x: x.update(independent=True),
                     lambda x: x['evaluation_seeds'].append(x['training_seeds'][0]),
                     lambda x: x.update(plan_quota=50),
                     lambda x: x.update(train_updates=0),
                     lambda x: x.update(action_candidates=[0,0,1,2])):
        bad=copy.deepcopy(m)
        mutation(bad)
        with pytest.raises(ValueError):validate(bad)


def test_real_observed_transition_proxy_preserves_negative_and_no_oracle():
    m=fixture()
    m['train_updates']=2
    m['evaluation_seeds']=[311]
    m['steps_per_evaluation_seed']=4
    result=run(m)
    assert result['training_rows']==288
    assert result['evaluation_rows']==4
    assert result['train_updates']==2
    assert validate_result(m,result)
    assert len(result['per_trial'])==4
    assert not result['independent'] and not result['acceptance_promoted']
    for modify in (lambda x: x['per_trial'].pop(),
                   lambda x: x['per_trial'].append(copy.deepcopy(x['per_trial'][0])),
                   lambda x: x['per_trial'][0].update(match=1),
                   lambda x: x['per_trial'][0].update(proposed_action=8),
                   lambda x: x.update(action_proxy_correct=-10)):
        bad=copy.deepcopy(result)
        modify(bad)
        with pytest.raises((ValueError,KeyError,TypeError)):
            validate_result(m,bad)


def test_generated_receipt_is_source_bound_after_ci_generation():
    import hashlib,os
    target=ROOT/'runs/model_plan_proxy_v1.json'
    if not target.exists():
        if os.environ.get('AURA_REQUIRE_MODEL_PROXY')=='1':
            pytest.fail('Required recorded proxy experiment is absent')
        pytest.skip('Public data generated separately after full suite')
    mraw=(ROOT/'docs/evaluation/model_plan_proxy_v1.json').read_bytes()
    result=json.loads(target.read_text())
    assert result['manifest_sha256']==hashlib.sha256(mraw).hexdigest()
    assert result['source_commit']
    assert validate_result(json.loads(mraw),result)
