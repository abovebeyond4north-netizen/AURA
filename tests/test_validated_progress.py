import numpy as np
import pytest
from aura.exploration.validated_progress import ValidatedProgressExplorer
from experiments.gridworld_exploration import NoisyTVWorld,run_one


def image(x=0):
    arr=np.full((3,7,7),x,dtype=np.uint8)
    return arr


def test_validation_bank_is_frozen_and_never_used_to_train():
    explorer=ValidatedProgressExplorer(seed=12,validation_per_action=2)
    initial=explorer.weights.copy()
    v1=explorer.observe(image(10),1,image(20))
    v2=explorer.observe(image(20),1,image(30))
    assert v1.role==v2.role=='validation'
    assert np.array_equal(explorer.weights,initial)
    frozen=[(x.copy(),y.copy()) for x,y in explorer.validation[1]]
    result=explorer.observe(image(30),1,image(40))
    assert result.role=='training'
    assert np.any(explorer.weights[1]!=initial[1])
    assert all(np.array_equal(x,a) and np.array_equal(y,b)
                   for (x,y),(a,b) in zip(frozen,explorer.validation[1]))
    assert explorer.training_count.sum()==1
    assert len(explorer.validation[1])==2


def test_actions_are_bounded_and_deterministic():
    a=ValidatedProgressExplorer(seed=44);b=ValidatedProgressExplorer(seed=44)
    for _ in range(30):
        x=a.choose(image(20));y=b.choose(image(20))
        assert x==y and 0<=x<5
        a.observe(image(20),x,image(10));b.observe(image(20),y,image(10))
    assert np.array_equal(a.weights,b.weights)
    with pytest.raises(ValueError):a.observe(image(20),5,image(10))
    with pytest.raises(ValueError):a.scores(image(20),'unknown')


def test_noisy_tv_image_is_observed_not_hidden_in_control_interface():
    e=NoisyTVWorld(diagnostics=True,noise=0,max_steps=10)
    o,_=e.reset(seed=11)
    b,_,_,_,_=e.step(5)
    assert b.shape==o.shape==(3,7,7)
    assert not np.array_equal(b[:,:3,:3],o[:,:3,:3])
    assert e.diagnostic_state()['agent']==(1,1)
    assert not e.diagnostic_state()['has_key']
    assert not e.diagnostic_state()['door_open']
    assert e.action_space.n==6


def test_matched_budget_and_observed_evidence():
    rows=[run_one(p,0,'noisy_tv',steps=40,max_episode_steps=20)
          for p in ('random','prediction_error','validated_progress')]
    assert all(r['interactions']==40 for r in rows)
    assert all(sum(r['agent_metrics']['action_counts'])==40 for r in rows)
    assert all(sum(r['agent_metrics']['training_updates_by_action']) +
                   sum(r['agent_metrics']['validation_examples_by_action'])==40 for r in rows)


def test_stable_visual_change_filters_one_off_random_colors():
    explorer=ValidatedProgressExplorer(seed=3)
    black=image(0)
    bright=black.copy()
    bright[:,:2,:2]=140
    for _ in range(12):
        explorer.observe(black,0,bright)
    stable=explorer._stable_effect(black,bright)
    unusual=black.copy()
    rng=np.random.default_rng(7)
    unusual[:,:2,:2]=rng.integers(1,255,size=(3,2,2),dtype=np.uint8)
    noisy=explorer._stable_effect(black,unusual)
    assert stable > .05
    assert noisy == 0


def test_noisy_tv_action_has_no_side_effect_on_underlying_mechanism():
    env=NoisyTVWorld(diagnostics=True,noise=0,max_steps=10)
    env.reset(seed=5)
    before=env.diagnostic_state().copy()
    for _ in range(3):
        env.step(5)
        assert env.diagnostic_state()==before
    # Original interaction is still available as action 4.
    assert env.action_space.contains(4)
    assert env.action_space.contains(5)
