from aura.exploration.hypothesis_memory import advance
from aura.environments.procedural_world import MOVES
M=(0,1,2,3)
def test_branch_evidence_is_private_and_persistent():
    base={}; p=(0,0); d=MOVES[0]; new={(2,2):'wall'}
    out=advance({(M,p,()):1.},0,{},new,base)
    assert len(out)==2 and base=={}
    assert abs(sum(out.values())-1)<1e-12
    for (_,q,memory) in out:
        assert dict(memory)[(q[0]+2,q[1]+2)]=='wall'
def test_historical_conflict_eliminates_branch():
    p=(0,0); d=MOVES[0]; base={d:'wall'}
    assert advance({(M,p,(((2,2),'wall'),)):1.},0,{}, {(2,2):'empty'},base)=={}
