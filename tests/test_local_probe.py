from aura.exploration.local_probe import local_probe
from aura.exploration.learned_dynamics import PERMUTATIONS
from aura.environments.procedural_world import MOVES

def test_local_safe_fallback_without_global_map():
    w={(m,(0,0)):1/24 for m in PERMUTATIONS}
    obs={d:'empty' for d in MOVES}
    assert local_probe(w,{},set(),obs) in range(4)
    assert local_probe(w,{},set(),{}) is None
    obs[MOVES[0]]='switch'
    assert local_probe(w,{},set(),obs) is None

def test_no_repeat_or_stationary_probe():
    w={(m,(0,0)):1/24 for m in PERMUTATIONS}
    assert local_probe(w,{},set(),{d:'wall' for d in MOVES}) is None
    assert local_probe(w,{}, {(frozenset(w),a) for a in range(4)}, {d:'empty' for d in MOVES}) is None
