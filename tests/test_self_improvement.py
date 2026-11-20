import pytest
from aura.config import Config
from aura.types import Modification
from aura.self_improvement.modification_registry import apply_modification
from aura.self_improvement.search_strategy import ImprovementSearch
from aura.self_improvement.proposer import ModificationProposer

def test_whitelist_and_forbidden_verifier():
    c=Config();mod=Modification('learning_rate','multiply',{'value':1.5},'v0')
    assert apply_modification(c,mod).lr>c.lr
    for forbidden in ('protected_verifier','acceptance_threshold','shell','arbitrary_code'):
        with pytest.raises(PermissionError):apply_modification(c,Modification(forbidden,'set',{'value':1},'v0'))

def test_adaptive_search_updates_and_bounded_proposal():
    s=ImprovementSearch();p=ModificationProposer(s)
    op,m=p.propose('v0',1)
    s.update(op,.1,15);assert s.counts[op]==1
    assert apply_modification(Config(),m)
