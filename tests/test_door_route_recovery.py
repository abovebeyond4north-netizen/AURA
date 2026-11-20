import numpy as np
import pytest
from aura.environments.procedural_world import COLORS
from aura.exploration.door_route_recovery import DoorRouteRecoveryCamera, CalibratedDoorRecoveryCamera
from aura.exploration.continuous_hazard import ContinuousHazardCamera
from aura.exploration.probabilistic_calibration import ProbabilisticCalibrationCamera
from aura.exploration.door_probe import Unmeasurable
from aura.exploration.learned_dynamics import PERMUTATIONS
from experiments.door_route_recovery_study import trial


def camera(tile):
    frame=np.tile(np.array(COLORS['empty'],dtype=np.uint8)[:,None,None],(1,7,7))
    frame[:,3,3]=COLORS['agent']
    c=DoorRouteRecoveryCamera(frame,lambda _:frame.copy())
    c.landmarks['door']=(1,0)
    c.map[(1,0)]=tile
    return c


def test_open_target_only_and_no_synthetic_observation():
    c=camera('empty'); target=(1,0); before=c.map.copy()
    assert target not in c.paths()
    def travel(route):
        assert c.paths()[target]==route
        c.position=target
    c.travel=travel
    c.recover_target(target)
    assert c.position==target and c.map==before
    assert c.landmarks['door']==target and c._door_recovery_target is None


@pytest.mark.parametrize('tile',['door','wall','unknown'])
def test_nonopen_door_not_unblocked(tile):
    c=camera(tile)
    with pytest.raises(Unmeasurable): c.recover_target((1,0))
    assert c.door_recoveries==0 and c.landmarks['door']==(1,0)


def test_failure_clears_override():
    c=camera('empty')
    def fail(route): raise Unmeasurable('test')
    c.travel=fail
    with pytest.raises(Unmeasurable): c.recover_target((1,0))
    assert c._door_recovery_target is None and (1,0) not in c.paths()


def test_combined_uses_probabilistic_calibration():
    assert CalibratedDoorRecoveryCamera.after_key_pickup is ProbabilisticCalibrationCamera.after_key_pickup


def test_matched_real_recovery_regression():
    args=(98178,9,3,PERMUTATIONS[2],PERMUTATIONS[2],0,.12)
    baseline=trial(*args,ContinuousHazardCamera)
    candidate=trial(*args,DoorRouteRecoveryCamera)
    assert not baseline['success'] and candidate['success']
    assert candidate['door_recoveries'] and not candidate['pose_errors']
