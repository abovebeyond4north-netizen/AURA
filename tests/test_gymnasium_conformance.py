"""Official Gymnasium checks; fallback tests cannot substitute for these."""
import os
import pytest

from aura.environments.procedural_world import HAS_GYMNASIUM, TASKS, create_environment


def test_backend_required_when_requested():
    if os.environ.get('AURA_REQUIRE_GYMNASIUM') == '1':
        assert HAS_GYMNASIUM, 'Official Gymnasium backend required by this run'


@pytest.mark.parametrize('task', TASKS)
def test_official_gymnasium_contract(task):
    if not HAS_GYMNASIUM:
        pytest.skip('Official Gymnasium unavailable; fallback is not conformance evidence')
    from gymnasium.utils.env_checker import check_env
    env = create_environment(task, noise=0.03)
    try:
        check_env(env, skip_render_check=False)
    finally:
        env.close()


def test_rgb_render_is_observation_consistent_and_rng_pure():
    import numpy as np
    for task in TASKS:
        a = create_environment(task, noise=0.2, render_mode='rgb_array')
        b = create_environment(task, noise=0.2, render_mode='rgb_array')
        try:
            observation, _ = a.reset(seed=8123)
            comparison, _ = b.reset(seed=8123)
            assert np.array_equal(observation, comparison)
            for _ in range(3):
                frame = a.render()
                assert frame.shape == (7, 7, 3) and frame.dtype == np.uint8
                assert np.array_equal(frame.transpose(2, 0, 1), observation)
                frame[:] = 0
            next_a, *_ = a.step(1)
            next_b, *_ = b.step(1)
            assert np.array_equal(next_a, next_b), task
        finally:
            a.close()
            b.close()


def test_render_mode_validated_and_optional():
    import pytest
    with pytest.raises(ValueError):
        create_environment(render_mode='invalid')
    env = create_environment(render_mode=None)
    try:
        assert env.render() is None
    finally:
        env.close()
