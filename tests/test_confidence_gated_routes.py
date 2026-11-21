from aura.environments.procedural_world import ProceduralWorld
from aura.exploration.confidence_gated_routes import ConfidenceGatedRouteCamera
from aura.exploration.efficient_routes import EfficientRouteCamera
from aura.exploration.persistent_pose import PersistentPoseCamera


def camera(size, col):
    env = ProceduralWorld(size=size,door_column=col,noise=0)
    obs,_ = env.reset(seed=12)
    return ConfidenceGatedRouteCamera(obs,lambda action: env.step(action)[0])


def test_gate_uses_visible_initial_landmarks_only():
    a = camera(9,2)
    b = camera(15,6)
    assert a.early_key_allowed
    assert not b.early_key_allowed
    assert isinstance(a,PersistentPoseCamera)
    assert ConfidenceGatedRouteCamera.prepare is not EfficientRouteCamera.prepare


def test_fallback_is_persistent_and_does_not_evaluate_hidden_size():
    import inspect
    source = inspect.getsource(ConfidenceGatedRouteCamera)
    assert 'self.landmarks' in source
    for forbidden in ('diagnostic_state','door_column','self.size','noise','_rng'):
        assert forbidden not in source
