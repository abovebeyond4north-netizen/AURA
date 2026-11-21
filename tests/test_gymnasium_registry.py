"""Check gym.make registration, Gymnasium EnvSpec recreation, and rendering."""
import pytest
from aura.environments.procedural_world import HAS_GYMNASIUM, TASKS


@pytest.mark.parametrize('task', TASKS)
def test_registered_world_gymnasium_factory_and_render(task):
    if not HAS_GYMNASIUM:
        pytest.skip('Gymnasium not installed; registration conformance unverified')
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env

    env_id = 'AURAOmega/ProceduralWorld-v0'
    if env_id not in gym.registry:
        gym.register(
            id=env_id,
            entry_point='aura.environments.procedural_world:ProceduralWorld',
        )

    wrapped = gym.make(env_id, task=task, noise=0.03, render_mode='rgb_array')
    try:
        env = wrapped.unwrapped
        assert env.spec is not None, 'Registered Gymnasium environment must have EnvSpec'
        assert env.spec.id == env_id
        check_env(env, skip_render_check=False)
        first, info = wrapped.reset(seed=71337)
        assert first.shape == (3,7,7) and info['task_family'] == task
        frame = wrapped.render()
        assert frame.shape == (7,7,3)
    finally:
        wrapped.close()
