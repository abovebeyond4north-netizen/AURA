import pytest
from aura.exploration.efficient_routes import EfficientRouteCamera
from aura.exploration.adaptive_routes import AdaptiveRouteCamera
from experiments.goal_acquisition import trial


def test_other_factorial_arms_are_rejected():
    c = object.__new__(EfficientRouteCamera)
    for arm in (0, 1, 3):
        with pytest.raises(ValueError, match='goal-only'):
            c.prepare(arm)


def test_clean_early_key_reduces_actions_under_both_control_mappings():
    for rev in (False, True):
        args = (88100, 11, 6, rev, 0, 'frontier')
        baseline = trial(*args, AdaptiveRouteCamera)
        candidate = trial(*args, EfficientRouteCamera)
        assert baseline['success'] and candidate['success']
        assert len(candidate['actions']) < len(baseline['actions']) <= 65


def test_verified_motion_unchanged():
    assert EfficientRouteCamera.act is AdaptiveRouteCamera.act
    assert EfficientRouteCamera.after_key_pickup is AdaptiveRouteCamera.after_key_pickup
