from pathlib import Path
import torch
from aura.agent import AuraOmega
from aura.config import Config

def test_temporal_heads_execute_during_normal_training(tmp_path):
    c=Config(steps=28,batch_size=6,warmup=6,max_episode_steps=12,
             latent_dim=8,hidden_dim=16,ensemble_size=2,log_dir=str(tmp_path),
             temporal_updates_per_episode=3)
    agent=AuraOmega(c)
    result=agent.train(task='key',condition='fixed')
    assert result['ledger_valid']
    assert result['temporal_training_steps']>=3
    obj=torch.load(Path(tmp_path)/'temporal.pt',weights_only=True)
    assert obj['update_steps']==result['temporal_training_steps']
    assert tuple(obj['horizons'])==(1,2,4)
    agent.close()
