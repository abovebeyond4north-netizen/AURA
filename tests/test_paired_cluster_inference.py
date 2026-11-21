import copy
import json
from pathlib import Path
import pytest
from tools.paired_cluster_inference import (paired_effect, _cluster_bootstrap, audit_public, audit_gate,
                                            PUBLIC_ARMS, GATED_ARMS)

ROOT = Path(__file__).resolve().parents[1]


def synthetic(n=4):
    rows=[]
    for seed in range(100,100+n):
        for rev in (False, True):
            for arm in PUBLIC_ARMS:
                rows.append({'seed':seed,'size':9,'column':2,'reverse':rev,'noise':.03,
                             'controller':arm,'success':arm=='PersistentPoseCamera',
                             'pose_errors':[],'actions':[0,4],'reason':None})
    return rows


def test_clusters_not_trials_are_resampled():
    out = paired_effect(synthetic(), 'PersistentPoseCamera', 'ComposedRouteCamera', PUBLIC_ARMS)
    assert out['pairs']==8
    assert out['paired']['baseline_only']==8
    assert out['paired']['candidate_only']==0
    assert len(out['per_seed'])==4
    assert out['exploratory_inference']['seed_clusters']==4
    assert out['exploratory_inference']['cluster_bootstrap_percentile_95']==[-1,-1]
    assert out['exploratory_inference']['sign_flip_two_sided_sensitivity']==.125
    assert out['exploratory_inference']['not_a_confirmatory_p_value']


def test_zero_effect_and_data_integrity():
    rows=synthetic()
    for row in rows: row['success'] = True
    x=paired_effect(rows,'PersistentPoseCamera','ComposedRouteCamera',PUBLIC_ARMS)
    assert x['exploratory_inference']['observed_mean_success_difference']==0
    assert x['exploratory_inference']['sign_flip_two_sided_sensitivity']==1
    for mutate in (lambda a:a.pop(),
                   lambda a:a.append(copy.deepcopy(a[0])),
                   lambda a:a[0].update(seed=9999),
                   lambda a:a[0].update(success=1),
                   lambda a:a[0].update(actions=[9]),
                   lambda a:a[0].update(pose_errors='not an audit')):
        bad=copy.deepcopy(rows)
        mutate(bad)
        with pytest.raises((ValueError,KeyError,TypeError)):
            paired_effect(bad,'PersistentPoseCamera','ComposedRouteCamera',PUBLIC_ARMS)


def test_rejects_unbalanced_cluster_conditions():
    rows=synthetic()
    one=copy.deepcopy(rows)
    for x in one:
        if x['seed']==100:
            x['noise'] = .1
    with pytest.raises(ValueError):
        paired_effect(one,'PersistentPoseCamera','ComposedRouteCamera',PUBLIC_ARMS)


def test_known_negative_regression_if_available():
    public=ROOT/'runs/public_holdout_v1.json'
    gate=ROOT/'runs/confidence_gate_public_v1.json'
    manifest=ROOT/'docs/evaluation/public_holdout_v1.json'
    if not (public.is_file() and gate.is_file() and manifest.is_file()):
        pytest.skip('Local captured study files unavailable; CI executes both reports explicitly')
    a=audit_public(manifest.read_bytes(),json.loads(public.read_text()))
    b=audit_gate(json.loads(gate.read_text()))
    assert a['comparisons'][0]['pairs']==72
    assert a['comparisons'][0]['paired']['net_candidate_minus_baseline']==-4
    assert b['comparisons'][0]['pairs']==72
    assert b['comparisons'][0]['paired']['net_candidate_minus_baseline']==5
    assert b['comparisons'][1]['paired']['net_candidate_minus_baseline']==-3


def test_sensitivity_small_seed_counts_not_certification():
    a=_cluster_bootstrap([-.2,-.1,0,.1])
    assert a['seed_clusters']==4
    assert a['independent_replications_insufficient']
    assert len(a['cluster_bootstrap_percentile_95'])==2
    assert not (a['sign_flip_two_sided_sensitivity'] is None)
