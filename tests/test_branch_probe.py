import pytest
from aura.exploration.branch_probe import expand,advance,select
from aura.exploration.door_probe import Unmeasurable
from aura.environments.procedural_world import MOVES
M=(0,1,2,3)
def test_expansion_fails_closed():
    with pytest.raises(Unmeasurable): expand({(M,(0,0),()):1},.03,cap=1)
def test_update_fails_closed():
    with pytest.raises(Unmeasurable): advance({(M,(0,0),()):1},0,{}, {},{},cap=1)
def test_private_known_route():
    d=MOVES[0]; b={(M,(0,0),((d,'empty'),)):1.}
    assert select(b,{},set())==0
    assert select(b,{}, {(frozenset(b),0)}) is None
