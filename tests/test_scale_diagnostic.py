from experiments.persistent_pose_study import trial
from aura.exploration.goal_search import RecoveryGoalCamera
from aura.exploration.persistent_pose import PersistentPoseCamera
from aura.exploration.composed_routes import ComposedRouteCamera


def test_large_layout_recovery_with_pose_audit():
    for size,col in ((13,4),(15,6)):
        for reverse in (False,True):
            for cls in (PersistentPoseCamera,ComposedRouteCamera):
                row=trial(97200,size,col,reverse,.03,cls)
                assert row['success'] and not row['pose_errors']
                assert len(row['actions'])<=65
