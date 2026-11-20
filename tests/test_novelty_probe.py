from experiments.novelty_probe_study import choose
from aura.environments.procedural_world import MOVES
M=(0,1,2,3)

def test_unknown_targets_and_used_actions_rejected():
    b={(M,(0,0),()):1.}
    assert choose(b,{},set()) is None
    assert choose(b,{MOVES[0]:'empty'},set())==0
    assert choose(b,{MOVES[0]:'empty'},{(frozenset(b),0)}) is None

def test_novelty_prefers_unmapped_view():
    b={(M,(0,0),()):1.}
    tiles={(r,c):'empty' for r in range(-4,0) for c in range(-3,4)}
    tiles[(1,0)]='empty'
    assert choose(b,tiles,set())==1
