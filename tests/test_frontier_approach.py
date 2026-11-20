import numpy as np
import pytest
from aura.environments.procedural_world import COLORS
from aura.exploration.door_probe import Unmeasurable
from aura.exploration.frontier_approach import FrontierApproachCamera, isolated_approaches
from aura.exploration.door_route_recovery import CalibratedDoorRecoveryCamera
from aura.exploration.learned_dynamics import PERMUTATIONS
from experiments.door_route_recovery_study import trial


def test_approach_requires_known_route_and_switch_isolation():
    door=(2,2); paths={(1,2):(1,), (3,2):(0,0)}
    assert isolated_approaches(door,(1,1),paths,len)==[(2,0,(0,0))]
    assert isolated_approaches(door,None,{},len)==[]


def test_no_unknown_route_and_bounded_abstention():
    frame=np.tile(np.array(COLORS['empty'],dtype=np.uint8)[:,None,None],(1,7,7)); frame[:,3,3]=COLORS['agent']
    def forbidden_step(_): pytest.fail('no safe route exists')
    camera=FrontierApproachCamera(frame,forbidden_step)
    camera.landmarks['door']=(20,20)
    camera.paths=lambda:{camera.position:()}
    before=camera.map.copy()
    with pytest.raises(Unmeasurable,match='bounded approach frontier exhausted'):
        camera.discover_approach()
    assert camera.map==before and camera.approach_views==0


def test_non_goal_arm_not_allowed():
    frame=np.tile(np.array(COLORS['empty'],dtype=np.uint8)[:,None,None],(1,7,7))
    camera=FrontierApproachCamera(frame,lambda _:pytest.fail('unexpected action'))
    with pytest.raises(ValueError): camera.prepare(0)


@pytest.mark.parametrize('seed,size,col,i,rate',[(98203,11,7,2,.12),(98203,11,7,19,0),(98205,11,8,2,.12)])
def test_real_missing_approach_recovery(seed,size,col,i,rate):
    args=(seed,size,col,PERMUTATIONS[i],PERMUTATIONS[i],.08,rate)
    old=trial(*args,CalibratedDoorRecoveryCamera,0)
    new=trial(*args,FrontierApproachCamera,0)
    assert not old['success'] and old['reason']=='no isolated approach to the door'
    assert new['success'] and not new['pose_errors']
    assert len(new['actions'])<=65
