"""The public paired audit must refuse incomplete or altered provenance."""
import copy
import json
from pathlib import Path
import pytest
from tools.navigation_audit import audit

ROOT = Path(__file__).resolve().parents[1]


def _fixture():
    m = json.loads((ROOT / 'docs/evaluation/public_holdout_v1.json').read_text())
    m['seeds'] = [42]
    m['layouts'] = [[9,2]]
    m['observation_noise'] = [0.03]
    from tools.public_navigation_study import CONTROLLERS, summarize
    rows = [{'seed':42,'size':9,'column':2,'reverse':rev,'noise':0.03,
             'controller':controller,'success':controller=='PersistentPoseCamera',
             'reason':'blocked','actions':[0,4], 'pose_errors':[]}
            for rev in (False,True) for controller in CONTROLLERS]
    import hashlib
    raw = json.dumps(m).encode()
    return raw, {'protocol':m['protocol'],'manifest_sha256':hashlib.sha256(raw).hexdigest(),
                 'rows':rows,'summary':summarize(rows,m)}


def test_negative_discordance_is_preserved_per_stratum():
    raw,r = _fixture()
    out = audit(raw, r)
    assert out['details']['overall']['paired_composed_vs_persistent'] == {
        'gains':0,'losses':2,'net':-2,'pairs':2}
    assert out['details']['by_seed']['42']['paired_composed_vs_persistent']['net'] == -2
    assert not out['independent'] and not out['acceptance_promoted']


def test_rejects_missing_duplicates_and_tampering():
    raw,r = _fixture()
    for mutation in (lambda x: x['rows'].pop(),
                     lambda x: x['rows'].append(copy.deepcopy(x['rows'][0])),
                     lambda x: x.update(manifest_sha256='0'*64),
                     lambda x: x['rows'][0].update(actions=[8]),
                     lambda x: x['rows'][0].update(success=1),
                     lambda x: x['summary'].update(total_trials=10)):
        candidate = copy.deepcopy(r)
        mutation(candidate)
        with pytest.raises((ValueError,KeyError,TypeError,AssertionError)):
            audit(raw, candidate)
