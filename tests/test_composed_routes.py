from aura.exploration.composed_routes import ComposedRouteCamera
from aura.exploration.efficient_routes import EfficientRouteCamera
from aura.exploration.persistent_pose import PersistentPoseCamera
from experiments.persistent_pose_study import trial


def test_composition_dispatch():
    assert ComposedRouteCamera.prepare is EfficientRouteCamera.prepare
    assert ComposedRouteCamera.act is PersistentPoseCamera.act
    assert ComposedRouteCamera.__init__ is PersistentPoseCamera.__init__


def test_composition_clean_audited():
    for reverse in (False, True):
        row=trial(92100,9,4,reverse,0,ComposedRouteCamera)
        assert row['success'] and not row['pose_errors']
        assert len(row['actions'])<=65
