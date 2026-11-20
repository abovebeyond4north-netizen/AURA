from aura.config import Config
from aura.types import Modification
from protected_verifier.version_manager import VersionManager
from protected_verifier.promotion import decide
import pytest

def test_version_promotion_rollback_requires_acceptance(tmp_path):
    manager=VersionManager(tmp_path,Config(lr=.001))
    modification=Modification('learning_rate','multiply',{'value':2.0},'v0')
    rejected=decide('v1','v0',[0.0]*6,[.1]*6,[.1]*6,{'steps':3})
    with pytest.raises(PermissionError):manager.promote(modification,rejected,[.1]*6,[.1]*6)
    approved=decide('v1','v0',[.01]*6,[.1]*6,[.09]*6,{'steps':3})
    assert approved.accepted
    manager.promote(modification,approved,[.1]*6,[.09]*6)
    assert manager.active_version=='v1' and manager.get_active().lr==.002
    manager.rollback('v0')
    assert manager.active_version=='v0' and manager.get_active().lr==.001


def test_forged_approval_does_not_bypass_policy(tmp_path):
    manager=VersionManager(tmp_path,Config())
    mod=Modification('learning_rate','multiply',{'value':1.2},'v0')
    forged=decide('v1','v0',[.01]*6,[.1]*6,[.09]*6,{'steps':6})
    assert forged.accepted
    with pytest.raises(PermissionError): manager.promote(mod,forged,[.1]*6,[.101]*6)


def test_version_identifiers_fail_closed_at_all_file_boundaries(tmp_path):
    import json
    manager = VersionManager(tmp_path / 'versions', Config())
    invalid = ('../escape', '..', '/tmp/escape', 'v/1', 'v\\1', '.hidden',
               'v\n1', '', 'v' * 129, 7, True)
    for version in invalid:
        with pytest.raises(ValueError):
            manager._write(version, Config())
        with pytest.raises(ValueError):
            manager.rollback(version)
    assert manager.active_version == 'v0'
    assert not (tmp_path / 'escape.json').exists()
    manager.active_file.write_text(json.dumps({'active_version': '../escape'}))
    with pytest.raises(ValueError):
        manager.get_active()


def test_version_promotion_rejects_unsafe_candidate_before_writing(tmp_path):
    manager = VersionManager(tmp_path, Config())
    mod = Modification('learning_rate', 'multiply', {'value': 1.2}, 'v0')
    approved = decide('../escape', 'v0', [.01] * 6, [.1] * 6, [.09] * 6, {'steps': 3})
    assert approved.accepted
    with pytest.raises(ValueError):
        manager.promote(mod, approved, [.1] * 6, [.09] * 6)
    assert manager.active_version == 'v0'
    assert not (tmp_path.parent / 'escape.json').exists()
