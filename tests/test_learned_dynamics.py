import numpy as np
import pytest
from aura.environments.procedural_world import COLORS
from aura.exploration.door_probe import Unmeasurable
from aura.exploration.learned_dynamics import (LearnedDynamicsCamera, initial_support, update_support, choose_experiment, PERMUTATIONS)
from experiments.learned_dynamics_study import trial


def test_prior_contains_every_bijection():
    assert len(initial_support((0,0))) == 24
    assert len({m for m,p in initial_support((0,0))}) == 24


def test_uninformative_evidence_does_not_select_mapping():
    s=update_support(initial_support((0,0)), 0, {}, {}, {})
    assert {m for m,p in s}==set(PERMUTATIONS)
    assert len(s)==48


def test_probe_excludes_unknown_and_dynamic_tiles():
    s=initial_support((0,0))
    assert choose_experiment(s,{},set()) is None
    assert choose_experiment(s,{(-1,0):'empty',(1,0):'empty',(0,1):'door',(0,-1):'empty'},set()) is None


def test_uniform_scene_does_not_commit():
    frame=np.tile(np.array(COLORS['empty'],dtype=np.uint8)[:,None,None],(1,7,7))
    frame[:,3,3]=COLORS['agent']
    c=LearnedDynamicsCamera(frame,lambda a:frame.copy())
    c.map.update({(r,k):'empty' for r in range(-5,6) for k in range(-5,6)})
    before=c.map.copy()
    with pytest.raises(Unmeasurable,match='learned action mapping unresolved'):
        c.after_key_pickup()
    assert c.map==before and c.position==(0,0) and c.mapping==(0,1,2,3)
    assert c.learning_probes<=c.max_calibration_probes


@pytest.mark.parametrize('mapping',[(2,0,3,1),(3,2,1,0),(0,1,2,3)])
def test_actual_world_mapping(mapping):
    row=trial(95100,9,5,mapping,.08,LearnedDynamicsCamera)
    assert row['completed_calibrations']==1
    assert row['mapping_errors']==0 and not row['pose_errors']
