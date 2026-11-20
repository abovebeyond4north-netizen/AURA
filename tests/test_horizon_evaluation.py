import numpy as np
from aura.config import Config
from aura.agent import AuraOmega
from aura.evaluation.runner import collect_development_trajectories,evaluate_temporal_heads

def test_development_horizon_data_and_inference(tmp_path):
    trajectories=collect_development_trajectories('switch',seed_index=4,episodes=2,max_steps=8)
    assert len(trajectories)==2
    assert all(len(obs)==len(act)+1 for obs,act,end in trajectories)
    a=AuraOmega(Config(steps=12,latent_dim=8,hidden_dim=16,ensemble_size=2,
                        batch_size=6,warmup=6,log_dir=str(tmp_path)))
    results=evaluate_temporal_heads(a.temporal_heads,'switch',seed_index=4,episodes=2)
    assert set(results)=={'h1_image_mse','h2_image_mse','h4_image_mse'}
    assert all(np.isfinite(v) for v in results.values())
    a.close()
