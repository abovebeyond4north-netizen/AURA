from aura.exploration.belief_lookahead import transition, contrast, lookahead
from aura.environments.procedural_world import MOVES

M=(0,1,2,3)
def test_unknown_or_dynamic_target_rejected():
    w={(M,(0,0)):1.}
    assert transition(w,{},0) is None
    assert transition(w,{},0,{MOVES[0]:'door'}) is None
    assert transition(w,{},0,{MOVES[0]:'wall'}) is None
    assert transition(w,{},0,{MOVES[0]:'empty'}) == {(M,MOVES[0]):1.}

def test_contrast_requires_observed_distinction():
    w={(M,(0,0)):.5,(M,(0,5)):.5}
    assert contrast(w,{})==0
    assert contrast(w,{(1,0):'wall',(1,5):'empty'})>0
    assert lookahead(w,{},set(),{},.03) is None
