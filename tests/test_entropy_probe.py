import pytest
from aura.exploration.entropy_probe import entropy,select_entropy
from aura.environments.procedural_world import MOVES
M=(0,1,2,3)
def test_entropy():
    assert entropy([1,1])==pytest.approx(1)
    assert entropy([1,0])==0
    assert entropy([0,0])==0

def test_safe_probe_and_repeat():
    b={(M,(0,0),((MOVES[0],'empty'),)):1.}
    assert select_entropy(b,{},set())==0
    assert select_entropy(b,{}, {(frozenset(b),0)}) is None
