"""Section 8 visual policy: observable motion, provenance, and real execution."""
import json
import numpy as np
import pytest
from aura.agent import AuraOmega
from aura.config import Config
from aura.environments.procedural_world import ProceduralWorld
from aura.exploration.visual_frontier import (
    VisualFrontierExplorer, visual_displacement,
)
from experiments.gridworld_exploration import NoisyTVWorld, run_one


def test_odometry_matches_real_observations_without_state_access():
    env=ProceduralWorld(task='key',noise=.05)
    obs,_=env.reset(seed=17)
    for action,expected in [(0,(0,0)),(2,(0,0)),(3,(0,1)),
                            (3,(0,1)),(1,(1,0)),(4,(0,0))]:
        nxt,*_=env.step(action)
        assert visual_displacement(obs,nxt)==expected
        obs=nxt
    with pytest.raises(ValueError):
        visual_displacement(obs[:,:6,:],obs)


def test_explorer_learns_inverted_actions_in_altered_environment():
    agent=VisualFrontierExplorer(seed=3,epsilon=0)
    env=ProceduralWorld(task='altered',noise=.03)
    obs,_=env.reset(seed=8)
    for _ in range(45):
        action=agent.choose(obs)
        nxt,_,done,trunc,_=env.step(action)
        agent.observe(obs,action,nxt)
        obs=nxt
        if done or trunc:break
    # The altered task maps action 0 to camera shift (+1, 0) and 2 to (0,+1).
    assert agent.action_shifts.get(0)==(1,0)
    assert agent.action_shifts.get(2)==(0,1)
    assert len(agent.visits)>3


def test_explorer_probes_non_navigation_action_then_avoids_noise():
    env=NoisyTVWorld(noise=0,max_steps=50)
    obs,_=env.reset(seed=3)
    agent=VisualFrontierExplorer(seed=101,n_actions=6,epsilon=0)
    for i in range(50):
        action=agent.choose(obs)
        nxt,_,done,trunc,_=env.step(action)
        agent.observe(obs,action,nxt)
        obs=nxt
        if done or trunc:break
    assert agent.action_counts[5]>=3  # actual exploration, not forbidden-action oracle
    assert agent.action_counts[5]<10
    assert agent.predictor.effect_observations[5]>=3


def test_mismatched_observation_and_action_are_rejected():
    agent=VisualFrontierExplorer(seed=3)
    obs=np.zeros((3,7,7),dtype=np.uint8)
    action=agent.choose(obs)
    with pytest.raises(ValueError):agent.observe(obs,(action+1)%5,obs)
    with pytest.raises(ValueError):agent.track_action(obs,4)
    with pytest.raises(ValueError):agent.observe(np.ones_like(obs),action,obs)


def test_external_action_keeps_pose_synchronized():
    env=ProceduralWorld(task='key',noise=0)
    obs,_=env.reset(seed=2)
    agent=VisualFrontierExplorer(seed=3)
    agent.track_action(obs,1)
    nxt,*_=env.step(1)
    agent.observe(obs,1,nxt)
    assert agent.pose==(1,0)
    assert agent.action_counts.sum()==1


def test_visual_frontier_real_benchmark_contract():
    result=run_one('visual_frontier',2,'noisy_tv',steps=70,max_episode_steps=35)
    assert result['interactions']==70
    assert sum(result['agent_metrics']['action_counts'])==70
    assert result['agent_metrics']['inferred_action_displacements']
    assert result['unique_physical_positions']<=result['unique_episode_positions']
    assert result['agent_metrics']['representation'].startswith('unlabeled')


def test_visual_frontier_runs_integrated_training_with_verified_ledger(tmp_path):
    config=Config(seed=28,steps=32,batch_size=8,warmup=8,
                  max_episode_steps=16,temporal_updates_per_episode=0,
                  curiosity_policy='visual_frontier',log_dir=str(tmp_path))
    agent=AuraOmega(config)
    try:
        result=agent.train(steps=32,condition='omega',task='key',log_every=16)
        assert result['ledger_valid']
        assert result['exploration_policy']=='visual_frontier'
        assert agent.visual_frontier.experiences==32
        assert agent.visual_frontier.action_counts.sum()==32
        logs=[json.loads(line) for line in (agent.output_dir/'metrics.jsonl').read_text().splitlines()]
        assert logs[-1]['exploration_metrics']['total_interactions']==32
        assert (agent.output_dir/'checkpoint.pt').exists()
    finally:agent.close()
