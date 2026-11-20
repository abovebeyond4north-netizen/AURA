import pytest
from experiments.doorway_curriculum import allocate
from experiments.doorway_study import TRAIN

def test_allocation_responds_to_failures_and_keeps_coverage():
    p={c:[{'success':c!=2}]*5 for c in TRAIN}
    fixed=allocate(p,False); adaptive=allocate(p,True)
    assert len(fixed)==len(adaptive)==9
    assert set(adaptive)==set(TRAIN)
    assert fixed.count(2)==3 and adaptive.count(2)>3
    p[2],p[6]=p[6],p[2]
    assert allocate(p,True).count(6)>3

def test_reserved_probe_rejected():
    with pytest.raises(ValueError): allocate({1:[{'success':False}]},True)
