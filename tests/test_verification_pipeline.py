import copy
import json
from pathlib import Path
import pytest
from tools.regenerate_ci_fixtures import FIXTURES, reproduce
from tools.public_navigation_study import CONTROLLERS, validate_manifest, summarize

ROOT = Path(__file__).resolve().parents[1]


def test_original_fixtures_are_reproducible():
    import hashlib
    for _, (expected, kind) in FIXTURES.items():
        assert hashlib.sha256(reproduce(kind).encode()).hexdigest() == expected


def test_public_manifest_rejects_mislabeling_and_overlap():
    m = json.loads((ROOT/'docs/evaluation/public_holdout_v1.json').read_text())
    validate_manifest(m)
    for change in (lambda x: x.update(independent=True),
                   lambda x: x['seeds'].append(x['seeds'][0]),
                   lambda x: x['controllers'].reverse(),
                   lambda x: x.update(action_cap=999)):
        bad = copy.deepcopy(m)
        change(bad)
        with pytest.raises(ValueError):
            validate_manifest(bad)


def test_paired_trial_integrity_and_negative_result():
    m = json.loads((ROOT/'docs/evaluation/public_holdout_v1.json').read_text())
    m['seeds'] = [2222]
    m['layouts'] = [[9,2]]
    m['observation_noise'] = [0.03]
    rows = [{"seed":2222,"size":9,"column":2,"reverse":rev,"noise":0.03,
             "controller":name,"success":name == "PersistentPoseCamera",
             "actions":[0,1,4],"pose_errors":[],"reason":"failure"}
            for rev in (False,True) for name in CONTROLLERS]
    r = summarize(rows,m)
    assert r['total_trials'] == 6
    assert r['paired_composed_vs_persistent']['regressed'] == 2
    assert r['paired_composed_vs_persistent']['net'] == -2
    assert not r['independent'] and not r['acceptance_promoted']
    with pytest.raises(ValueError): summarize(rows[:-1],m)
    with pytest.raises(ValueError): summarize(rows+[rows[0]],m)
