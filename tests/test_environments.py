import numpy as np
import pytest
from aura.environments.procedural_world import create_environment,TASKS
from aura.environments.task_generator import task_seed

def test_environment_reproducible_and_invalid_action():
    a=create_environment('key');b=create_environment('key')
    x,_=a.reset(seed=42);y,_=b.reset(seed=42)
    assert np.array_equal(x,y) and a.observation_space.contains(x)
    for action in [0,4,3,1]:
        x,r,d,t,info=a.step(action);y,rr,dd,tt,ii=b.step(action)
        assert np.array_equal(x,y) and r==rr and d==dd and t==tt
    with pytest.raises(ValueError): a.step(5)
    with pytest.raises(PermissionError):a.diagnostic_state()

def test_families_and_noise():
    for t in TASKS:
        env=create_environment(task=t);obs,_=env.reset(seed=4)
        assert obs.shape==(3,7,7)
    env=create_environment('hidden',diagnostics=True)
    _,info=env.reset(seed=4)
    assert 'rule' not in info and 'rule' in env.diagnostic_state()
    env=create_environment('noise')
    x,_=env.reset(seed=4);y,_=env.reset(seed=5)
    assert not np.array_equal(x,y)

def test_seed_splits():
    assert len({task_seed(s,5) for s in ('train','dev','protected')})==3


def test_key_switch_and_hidden_rules_permit_actual_goal_reach():
    # White-box reference controls for environment validity; not supplied to learner.
    for task,opening in [('key','key'),('noise','key'),('switch','switch'),('hidden','key')]:
        env=create_environment(task,noise=0)
        env.reset(seed=0)
        if task=='hidden': env._rule=0
        if opening=='key':
            env.step(1)  # collect key at (2,1)
            for _ in range(3):env.step(3)
            env.step(1)  # (3,4) adjacent door
            env.step(4)
        else:
            env.step(3);env.step(4) # activate switch at (1,2)
            for _ in range(2):env.step(3)
            for _ in range(2):env.step(1) # (3,4)
        assert env.door_open,task
        for _ in range(4): obs,r,done,trunc,info=env.step(1)
        for _ in range(3): obs,r,done,trunc,info=env.step(3)
        assert done and info['success'],task

def test_hidden_switch_rule_can_open_barrier():
    env=create_environment('hidden',diagnostics=True)
    env.reset(seed=0)
    env._rule=1  # evaluator-controlled known mechanism fixture
    env.step(3)
    env.step(4)
    assert env.door_open and env.diagnostic_state()['switch_on']
