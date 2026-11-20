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
        check_env(env, skip_render_check=True)
    finally:
        env.close()
