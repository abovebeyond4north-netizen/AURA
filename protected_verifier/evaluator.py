"""Train baseline and candidate under matched budgets on fixed random policies.
The verifier is executed in a separate subprocess by the CLI; isolation from
arbitrary hostile local code is NOT guaranteed by Python subprocess alone.
"""
from __future__ import annotations
import dataclasses
import numpy as np
from aura.config import Config,seed_everything
from aura.world_model.dynamics import WorldModel
from aura.environments.procedural_world import create_environment
from aura.environments.task_generator import task_seed
from aura.self_improvement.modification_registry import apply_modification
from aura.types import Modification
from .promotion import decide
from .resource_limits import validate_budget
from .test_registry import protected_seed,PROTECTED_TASKS

def collect(task,seed,steps):
    env=create_environment(task=task,max_steps=40,noise=.03 if task!='noise' else .5)
    obs,_=env.reset(seed=seed)
    rng=np.random.default_rng(seed+9); x=[]; a=[]; y=[]
    for _ in range(steps):
        action=int(rng.integers(5)); next_obs,_,done,trunc,_=env.step(action)
        x.append(obs.copy()); y.append(next_obs.copy()); a.append(action)
        obs=next_obs
        if done or trunc: obs,_=env.reset(seed=int(rng.integers(1,2**30)))
    return np.stack(x),np.array(a),np.stack(y)

def compare(config:Config, modification:Modification,steps=60,replicates=5,parent_version='v0',candidate_version='v1'):
    validate_budget(steps,replicates)
    if modification.parent_version!=parent_version: raise ValueError('Parent mismatch')
    candidate_config=apply_modification(config,modification)
    old=[];new=[]
    for i in range(replicates):
        seed=task_seed('train',1000+i)
        task=PROTECTED_TASKS[i%len(PROTECTED_TASKS)]
        training=collect(task,seed,max(steps+20,40))
        heldout=collect(task,protected_seed(i),32)
        run_losses=[]
        for conf in (config,candidate_config):
            seed_everything(conf.seed+i)
            model=WorldModel(conf.latent_dim,conf.hidden_dim,conf.ensemble_size,conf.lr,'cpu')
            rng=np.random.default_rng(seed+17)
            for _ in range(steps):
                indices=rng.integers(0,len(training[0]),size=min(conf.batch_size,len(training[0])))
                batch=tuple(arr[indices] for arr in training)
                model.train_batch(batch)
            run_losses.append(model.prediction_loss(heldout))
        old.append(run_losses[0]);new.append(run_losses[1])
    return decide(candidate_version,parent_version,[b-c for b,c in zip(old,new)],old,new,
                  {'trained_model_steps':2*replicates*steps,'replicates':replicates})
