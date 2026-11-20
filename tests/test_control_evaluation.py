import pytest
import torch
from aura.agent import AuraOmega
from aura.config import Config
from aura.evaluation.control import evaluate_control
from aura.environments.task_generator import task_seed


def test_control_comparator_is_paired_bounded_and_nontraining(tmp_path):
    agent=AuraOmega(Config(steps=12,warmup=4,batch_size=4,latent_dim=8,hidden_dim=16,ensemble_size=2,log_dir=str(tmp_path)))
    try:
        agent.train(task='key',condition='fixed')
        state={k:v.detach().clone() for k,v in agent.actor.state_dict().items()}
        world_state={k:v.detach().clone() for k,v in agent.world_model.state_dict().items()}
        steps=agent.actor.train_steps
        r=evaluate_control(agent.world_model,agent.actor,tasks=('key','noise'),episodes=2,max_steps=6)
        assert len(r['rows'])==8
        for task in r['summary']:
            seeds=[[a['seed'] for a in r['rows'] if a['task']==task and a['policy']==p] for p in ('actor','random')]
            assert seeds[0]==seeds[1] and len(set(seeds[0]))==2
            assert task_seed('dev',agent.config.seed) not in seeds[0]
        assert all(1<=a['interactions']<=6 for a in r['rows'])
        assert all(torch.equal(v,agent.actor.state_dict()[k]) for k,v in state.items())
        assert all(torch.equal(v,agent.world_model.state_dict()[k]) for k,v in world_state.items())
        assert steps==agent.actor.train_steps and len(agent.ledger.records('experience'))==12
        assert r==evaluate_control(agent.world_model,agent.actor,tasks=('key','noise'),episodes=2,max_steps=6)
        with pytest.raises(ValueError):evaluate_control(agent.world_model,agent.actor,episodes=0)
    finally:agent.close()
