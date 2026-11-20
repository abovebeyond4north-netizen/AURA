import numpy as np
import pytest
from aura.environments.procedural_world import COLORS
from aura.exploration.door_probe import Unmeasurable
from aura.exploration.change_dynamics import ChangeDynamicsCamera, ReplanningChangeCamera, deterministic_support
from aura.exploration.context_dynamics import ContextDynamicsCamera
from experiments.change_dynamics_study import trial


def test_known_wall_and_free_transition_constraints():
    state={((0,1,2,3),(0,0))}
    assert deterministic_support(state,0,{}, {},{(-1,0):'wall'}) == state
    assert deterministic_support(state,0,{}, {},{(-1,0):'empty'}) == {((0,1,2,3),(-1,0))}
    assert len(deterministic_support(state,0,{}, {},{})) == 2


def test_ambiguous_motion_fails_closed():
    frame=np.tile(np.array(COLORS['empty'],dtype=np.uint8)[:,None,None],(1,7,7))
    frame[:,3,3]=COLORS['agent']
    camera=ChangeDynamicsCamera(frame,lambda _:frame.copy())
    camera.monitoring=True
    camera.map.update({(r,c):'empty' for r in range(-6,7) for c in range(-6,7)})
    prior=camera.map.copy()
    with pytest.raises(Unmeasurable): camera.act(0,(-1,0))
    assert camera.map==prior and camera.mapping==(0,1,2,3) and camera.position==(0,0)
    assert camera.change_events==1 and camera.change_resolutions==0
    assert camera.change_probes<=camera.max_calibration_probes


@pytest.mark.parametrize('cls',[ChangeDynamicsCamera,ReplanningChangeCamera])
def test_door_change_detected_without_hook(cls):
    row=trial(97000,9,5,(0,1,2,3),(1,3,2,0),0,'door',cls)
    assert row['success'] and row['change_resolutions']>0
    assert row['context_probes']==0 and row['mapping_errors']==0 and not row['pose_errors']
    assert len(row['actions'])<=65


def test_no_change_does_not_force_reidentification():
    row=trial(97000,9,5,(0,1,2,3),(0,1,2,3),0,'none',ChangeDynamicsCamera)
    assert row['success'] and row['change_events']==0 and row['context_probes']==0
    assert not row['pose_errors']


def test_late_change_detected_without_door_notification():
    row=trial(97000,9,5,(2,3,0,1),(0,2,3,1),0,'late',ChangeDynamicsCamera)
    assert row['success'] and row['change_resolutions']>0 and row['mapping_errors']==0
    assert row['context_probes']==0 and not row['pose_errors']
