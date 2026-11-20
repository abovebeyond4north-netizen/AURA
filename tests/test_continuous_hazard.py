import pytest
from aura.exploration.continuous_hazard import propagate
from aura.exploration.learned_dynamics import PERMUTATIONS

def test_reset_preserves_pose_mass_and_restores_mapping_support():
    a,b=PERMUTATIONS[:2]; p,q=(0,0),(1,0)
    w=propagate({(a,p):.75,(b,q):.25},.12)
    assert len(w)==48 and min(w.values())>0
    assert sum(v for (_,x),v in w.items() if x==p)==pytest.approx(.75)
    assert w[a,p]==pytest.approx(.75*(.88+.12/24))
    assert w[b,p]==pytest.approx(.75*.12/24)

def test_reset_no_fabricated_pose_or_empty_support():
    assert propagate({},.12)=={}
    with pytest.raises(ValueError): propagate({},0)
    with pytest.raises(ValueError): propagate({},1.1)

def test_ambiguous_camera_never_commits_pose():
    import numpy as np
    from aura.environments.procedural_world import COLORS
    from aura.exploration.continuous_hazard import ContinuousHazardCamera
    from aura.exploration.door_probe import Unmeasurable
    frame=np.tile(np.array(COLORS['empty'],dtype=np.uint8)[:,None,None],(1,7,7))
    frame[:,3,3]=COLORS['agent']
    cam=ContinuousHazardCamera(frame,lambda _:frame.copy())
    cam.monitoring=True
    cam.map.update({(r,c):'empty' for r in range(-6,7) for c in range(-6,7)})
    before=cam.map.copy()
    with pytest.raises(Unmeasurable): cam.act(0,(-1,0))
    assert cam.map==before and cam.position==(0,0) and cam.posterior_updates==0
