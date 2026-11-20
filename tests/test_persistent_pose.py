import numpy as np
import pytest
from aura.environments.procedural_world import COLORS
from aura.exploration.door_probe import Unmeasurable
from aura.exploration.persistent_pose import PersistentPoseCamera, compatible, propagate, select_probe, view
from aura.exploration.adaptive_routes import AdaptiveRouteCamera
from experiments.persistent_pose_study import trial


def test_uniform_frames_keep_both_motion_hypotheses():
    tiles={(r,c):'empty' for r in range(-5,6) for c in range(-5,6)}
    frame=np.tile(np.array(COLORS['empty'],dtype=np.uint8)[:,None,None],(1,7,7))
    old=view(frame)
    assert propagate({(0,0)},(0,1),old,old,tiles)=={(0,0),(0,1)}
    assert propagate({(0,0),(0,1)},(0,1),old,old,tiles)=={(0,0),(0,1),(0,2)}


def test_insufficient_evidence_does_not_remove_hypothesis():
    assert compatible({}, {}, (0,0),(0,1), {})
    assert propagate({(0,0)},(0,1),{},{},{})=={(0,0),(0,1)}


def test_static_map_distinguishes_two_hypotheses():
    tiles={(0,2):'wall',(0,3):'empty'}
    assert propagate({(0,0)},(0,1),{}, {(0,2):'empty'},tiles)=={(0,1)}


def test_probe_avoids_unknown_and_landmark_targets():
    assert select_probe({(0,0),(0,1)}, {},set()) is None
    tiles={(1,0):'empty',(1,1):'key'}
    assert select_probe({(0,0),(0,1)}, tiles,set()) is None
    tiles[(1,1)]='empty'
    assert select_probe({(0,0),(0,1)},tiles,set())==1


def test_recovery_audit_checks_real_pose():
    for rev in (False,True):
        row=trial(90200,15,4,rev,.03,PersistentPoseCamera)
        assert row['recoveries']>0 and row['resolutions']>0
        assert row['max_support']>=2
        assert not row['pose_errors'] and len(row['actions'])<=65


def test_clean_small_layout_no_regression():
    for cls in (AdaptiveRouteCamera,PersistentPoseCamera):
        row=trial(90200,9,6,False,0,cls)
        assert row['success'] and not row['pose_errors']


def test_unresolved_recovery_does_not_write_map():
    frame=np.tile(np.array(COLORS['empty'],dtype=np.uint8)[:,None,None],(1,7,7))
    frame[:,3,3]=COLORS['agent']
    camera=PersistentPoseCamera(frame,lambda action:frame.copy())
    camera.map.update({(r,c):'empty' for r in range(-7,8) for c in range(-7,8)})
    snapshot=camera.map.copy()
    with pytest.raises(Unmeasurable,match='persistent pose unresolved'):
        camera.act(3,(0,1))
    assert camera.position==(0,0) and camera.map==snapshot
    assert 0<camera.probe_actions<=camera.max_probes
