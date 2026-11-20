import numpy as np
import pytest
from aura.exploration.goal_search import stationary_shift, GoalCamera
from experiments.goal_acquisition import trial


def test_uniform_views_abstain():
    a = np.zeros((3,7,7),dtype=np.uint8)
    assert stationary_shift(a,a) is None


def test_stationary_translation_ignores_center():
    rng = np.random.default_rng(4)
    scene = rng.integers(0,256,(3,7,8),dtype=np.uint8)
    a,b = scene[:,:,:7].copy(),scene[:,:,1:].copy()
    a[:,3,3] = b[:,3,3] = 255
    assert stationary_shift(a,b) == (0,1)


@pytest.mark.parametrize('reverse',[False,True])
def test_clean_failed_geometry(reverse):
    out = trial(72100,11,4,reverse,0,'frontier',GoalCamera)
    assert out['success'] and len(out['actions']) <= 65
