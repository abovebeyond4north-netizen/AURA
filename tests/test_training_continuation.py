"""Live in-process continuation and standalone actor checkpoint evidence."""
import numpy as np
import pytest
import torch
from pathlib import Path
from aura.agent import AuraOmega
from aura.config import Config
from aura.planning.actor import Actor


def test_segmented_training_matches_uninterrupted_training(tmp_path):
    common=dict(seed=21,steps=24,max_episode_steps=8,batch_size=4,warmup=4,latent_dim=8,hidden_dim=16,ensemble_size=2,temporal_updates_per_episode=2)
    def run(folder,segments):
        agent=AuraOmega(Config(**common,log_dir=str(folder)))
        try:
            for target in segments: report=agent.train(steps=target,task='key',condition='fixed')
            rows=[(r['payload']['episode_id'],r['payload']['seed'],r['payload']['action'],r['payload']['reward'],r['payload']['observation']['sha256']) for r in agent.ledger.records('experience')]
            return report,rows,{k:v.detach().clone() for k,v in agent.actor.state_dict().items()}
        finally:agent.close()
    straight,rows1,state1=run(tmp_path/'straight',[24])
    resumed,rows2,state2=run(tmp_path/'resumed',[5,13,24])
    assert rows1==rows2
    assert straight['episodes']==resumed['episodes']==3
    assert straight['temporal_training_steps']==resumed['temporal_training_steps']
    assert straight['resource_metrics']['optimization_or_concept_steps']==resumed['resource_metrics']['optimization_or_concept_steps']
    assert all(torch.equal(v,state2[k]) for k,v in state1.items())
    assert resumed['ledger_valid']


def test_actor_checkpoint_roundtrip_and_policy_match(tmp_path):
    agent=AuraOmega(Config(steps=12,max_episode_steps=6,batch_size=4,warmup=4,latent_dim=8,hidden_dim=16,ensemble_size=2,log_dir=str(tmp_path)))
    try:
        report=agent.train(task='key',condition='fixed')
        path=Path(report['actor_checkpoint'])
        assert path.is_file()
        actor=Actor(8)
        assert actor.restore(path)==agent.actor.train_steps>0
        obs=agent.replay.data[0].observation
        assert actor.select(agent.world_model,obs,0.,np.random.default_rng(0))==agent.actor.select(agent.world_model,obs,0.,np.random.default_rng(1))
        assert all(torch.equal(v,actor.state_dict()[k]) for k,v in agent.actor.state_dict().items())
    finally:agent.close()
