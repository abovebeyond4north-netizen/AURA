import numpy as np
import pytest
from aura.exploration.reward_routes import RewardRouteExplorer
from experiments.reward_route_study import run_one


def test_routes_require_actual_positive_terminal_reward_and_monitor_replay():
    x=np.zeros((3,7,7),dtype=np.uint8); y=np.full_like(x,123)
    p=RewardRouteExplorer(seed=1,retry_probability=1.)
    a=p.choose(x); p.observe(x,a,x,reward=0.,terminated=False,truncated=True)
    assert not p.routes
    p.reset_episode(); a=p.choose(x); p.observe(x,a,x,reward=1.,terminated=True,truncated=False,evidence_id='observed')
    assert len(p.routes)==1 and p.routes[0].evidence_ids==('observed',)
    p.reset_episode(); assert p.choose(x)==a
    p.observe(x,a,x,reward=1.,terminated=True,truncated=False)
    assert p.metrics()['exact_successful_reproductions']==1
    p.reset_episode(); p.choose(y)
    assert p.active is None


def test_route_retry_aborts_on_changed_intermediate_observation():
    x=np.zeros((3,7,7),dtype=np.uint8); y=np.full_like(x,123)
    p=RewardRouteExplorer(seed=2,retry_probability=1.)
    a=p.choose(x);p.observe(x,a,x,reward=-.01,terminated=False,truncated=False)
    b=p.choose(x);p.observe(x,b,x,reward=1.,terminated=True,truncated=False)
    p.reset_episode(); assert p.choose(x)==a
    p.observe(x,a,y,reward=-.01,terminated=False,truncated=False)
    p.choose(y)
    assert p.active is None and p.aborted==1
    assert p.reproduced==0


def test_real_environment_route_study_is_budgeted():
    r=run_one('reward_routes','key',82,steps=320,horizon=80)
    assert r['interactions']==r['metrics']['interactions']==320
    assert r['metrics']['candidate_routes_discovered']>=1
    assert r['successes']>=1


def test_route_policy_runs_inside_actual_agent_with_observed_ledger(tmp_path):
    from aura.agent import AuraOmega
    from aura.config import Config
    agent=AuraOmega(Config(seed=1082,steps=320,max_episode_steps=80,warmup=4,batch_size=4,latent_dim=8,hidden_dim=16,ensemble_size=2,temporal_updates_per_episode=0,curiosity_policy='reward_routes',log_dir=str(tmp_path)))
    try:
        with pytest.raises(ValueError):agent.train(task='key',condition='omega')
        r=agent.train(task='key',condition='curiosity')
        assert r['steps']==320 and r['ledger_valid']
        assert r['exploration_metrics']['interactions']==320
        refs={record['record_id'] for record in agent.ledger.records('experience')}
        assert len(refs)==320
        for route in r['exploration_metrics']['retained_routes']:
            assert len(route['evidence_ids'])==route['length']
            assert set(route['evidence_ids'])<=refs
        assert r['exploration_metrics']['candidate_routes_discovered']>=1
    finally:agent.close()
