"""Fixed held-out image prediction evaluation without changing model weights."""
from protected_verifier.evaluator import collect
from aura.environments.task_generator import task_seed

def evaluate_world_model(world,task='key',seed_index=0,transitions=60):
    data=collect(task,task_seed('dev',seed_index),transitions)
    return {'heldout_image_mse':world.prediction_loss(data),'transitions':transitions}


def collect_development_trajectories(task='key', seed_index=0, episodes=3, max_steps=12):
    """Fresh development episodes, never protected-test seeds; observed data only."""
    import numpy as np
    from aura.environments.procedural_world import create_environment
    trajectories=[]
    for ep in range(episodes):
        seed=task_seed('dev',10000+seed_index*100+ep)
        env=create_environment(task,max_steps=max_steps,noise=.03 if task!='noise' else .5)
        obs,_=env.reset(seed=seed)
        rng=np.random.default_rng(seed+55)
        obs_list=[obs.copy()];actions=[];endings=[]
        for _ in range(max_steps):
            action=int(rng.integers(5))
            nxt,_,done,trunc,_=env.step(action)
            actions.append(action);obs_list.append(nxt.copy());endings.append(done or trunc)
            if done or trunc: break
        trajectories.append((np.stack(obs_list),actions,endings))
    return trajectories


def evaluate_temporal_heads(heads,task='key',seed_index=0,episodes=3):
    """Report action-conditioned direct next-image MSE at distinct horizons."""
    return {f'h{h}_image_mse':m for h,m in
            heads.evaluate(collect_development_trajectories(task,seed_index,episodes)).items()}


def evaluate_temporal_latent_heads(heads,task='key',seed_index=0,episodes=3):
    """Held-out action-sequence conditioned latent-state MSE by horizon."""
    return {f'h{h}_latent_mse':loss for h,loss in
            heads.evaluate_latent(collect_development_trajectories(task,seed_index,episodes)).items()}
