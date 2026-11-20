from experiments.two_step_probe_study import partition, residual, choose
from aura.environments.procedural_world import MOVES
M=(0,1,2,3)
def test_unknown_target_rejected():
    b={(M,(0,0),()):1.}
    assert partition(b,{},0) is None
    assert choose(b,{},set(),0) is None

def test_single_pose_has_zero_residual():
    b={(M,(0,0),()):1.}
    g=partition(b,{MOVES[0]:'empty'},0)
    assert residual(g)==0
    assert choose(b,{MOVES[0]:'empty'},set(),0)==0
    assert choose(b,{MOVES[0]:'empty'},{(frozenset(b),0)},0) is None
