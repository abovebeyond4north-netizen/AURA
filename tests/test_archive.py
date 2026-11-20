from aura.types import Modification
from aura.self_improvement.candidate_archive import CandidateArchive
from protected_verifier.promotion import decide
import pytest

def test_archive_immutability_lineage(tmp_path):
    a=CandidateArchive(tmp_path/'lineage.jsonl')
    report=decide('v1','v0',[0]*5,[.1]*5,[.1]*5,{'steps':30})
    mod=Modification('learning_rate','multiply',{'value':1.5},'v0')
    a.record('v1','v0',mod,report)
    with pytest.raises(ValueError):a.record('v1','v0',mod,report)
    with pytest.raises(ValueError):a.record('v2','nonexistent',mod,report)
    assert CandidateArchive(tmp_path/'lineage.jsonl').records()[0]['status']=='ARCHIVED'
