import numpy as np
import pytest
from aura.environments.procedural_world import COLORS
from aura.exploration.door_probe import Unmeasurable
from aura.exploration.hazard_dynamics import HazardDynamicsCamera, hazard_prior, update, qualified_probe
from aura.exploration.learned_dynamics import PERMUTATIONS
from experiments.change_dynamics_study import trial


def test_positive_hazard_preserves_all_mappings():
    mapping=(0,1,2,3)
    w=hazard_prior({mapping:1.},(0,0),.03)
    assert len(w)==24 and all(v>0 for v in w.values())
    assert sum(w.values())==pytest.approx(1.)
    assert w[mapping,(0,0)]==pytest.approx(.97+.03/24)
    with pytest.raises(ValueError): hazard_prior({mapping:1.},(0,0),0.)


def test_filter_preserves_possible_outcomes_not_only_most_likely():
    w=hazard_prior({PERMUTATIONS[0]:1.},(0,0),.03)
    post=update(w,0,{},{},{})
    assert len(post)==48 and sum(post.values())==pytest.approx(1.)
    assert all(v>0 for v in post.values())
    assert qualified_probe(post,{},set()) is None


def test_ambiguous_observations_do_not_commit_position_or_mapping():
    frame=np.tile(np.array(COLORS['empty'],dtype=np.uint8)[:,None,None],(1,7,7))
    frame[:,3,3]=COLORS['agent']
    cam=HazardDynamicsCamera(frame,lambda _:frame.copy())
    cam.monitoring=True
    cam.map.update({(r,c):'empty' for r in range(-6,7) for c in range(-6,7)})
    before=cam.map.copy()
    with pytest.raises(Unmeasurable): cam.act(0,(-1,0))
    assert cam.map==before and cam.position==(0,0) and cam.mapping==(0,1,2,3)
    assert cam.change_probes<=cam.max_calibration_probes and cam.posterior_updates==0

@pytest.mark.parametrize('when,initial,after', [('none',(0,1,2,3),(0,1,2,3)),('door',(0,1,2,3),(1,3,2,0)),('late',(2,3,0,1),(0,2,3,1))])
def test_matched_public_fixture(when,initial,after):
    row=trial(97000,9,5,initial,after,0,when,HazardDynamicsCamera)
    assert row['success'] and not row['pose_errors'] and not row['mapping_errors']
    assert row['context_probes']==0 and len(row['actions'])<=65
    if when!='none': assert row['change_resolutions']>0
