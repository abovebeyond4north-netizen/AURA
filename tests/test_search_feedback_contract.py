import math
import pytest
from aura.self_improvement.search_strategy import ImprovementSearch
from aura.self_improvement.proposer import ModificationProposer

@pytest.mark.parametrize('gain,cost', [(math.nan,1),(math.inf,1),(-math.inf,1),(0,math.nan),(0,math.inf),(0,-1),(True,1),(0,True),(1e308,1e308)])
def test_invalid_or_overflow_feedback_is_atomic(gain,cost):
    s=ImprovementSearch(); op=s.select()
    if gain==1e308:
        s.update(op,gain,0)
    before=(s.counts.copy(),s.scores.copy())
    with pytest.raises(ValueError): s.update(op,gain,cost)
    assert (s.counts,s.scores)==before

@pytest.mark.parametrize('weakness',[math.nan,math.inf,-math.inf,True])
def test_invalid_weakness(weakness):
    with pytest.raises(ValueError): ModificationProposer(ImprovementSearch()).propose('v0',weakness)

def test_negative_finite_gain_is_retained():
    s=ImprovementSearch(); op=s.select(); s.update(op,-.5,100)
    assert s.counts[op]==1 and s.scores[op]==pytest.approx(-.6)
