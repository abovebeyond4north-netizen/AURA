"""Noise-aware curiosity based on validated learning progress and epistemic proxy.

Uncertainty is NOT calibrated probability. A fixed validation batch must be used
for before/after comparisons. Independent irreducible image noise is penalized.
"""
from __future__ import annotations
import numpy as np

class LearningProgress:
    def __init__(self,window=12):
        self.window=window; self.errors=[]
    def update(self, fixed_validation_mse: float) -> float:
        if not np.isfinite(fixed_validation_mse): raise ValueError('Non-finite validation error')
        past = np.mean(self.errors[-self.window:]) if len(self.errors)>=self.window else None
        self.errors.append(float(fixed_validation_mse))
        return max(0.0, float(past-fixed_validation_mse)) if past is not None else 0.0

class Curiosity:
    def __init__(self,world, noise_penalty=4.0, diversity=0.05, cost=0.001):
        self.world=world; self.noise_penalty=noise_penalty
        self.diversity=diversity; self.cost=cost; self.counts=np.ones(5)
        self.progress=LearningProgress(); self.validated_progress=0.
        self.irreducible_error=0.
    def update_validation(self,loss):
        self.validated_progress=self.progress.update(loss)
        self.irreducible_error=.9*self.irreducible_error+.1*float(loss)
    def score(self, epistemic, action, noise_estimate=0.0, transferable_benefit=0.0):
        # Noise floor attenuates epistemic enthusiasm for uncontrollable stimulus.
        ig = max(0,float(epistemic)) / (1+self.noise_penalty*max(0,float(noise_estimate)))
        progress = self.validated_progress/(1+max(0,float(noise_estimate)))
        return ig+progress+float(transferable_benefit)+self.diversity/np.sqrt(self.counts[action])-self.cost
    def choose(self,observation,epsilon=0.15,rng=None):
        rng=rng or np.random.default_rng()
        if rng.random()<epsilon: action=int(rng.integers(5))
        else:
            batch=np.repeat(observation[None],5,axis=0)
            with __import__('torch').no_grad():
                u=self.world.estimate_uncertainty(batch,np.arange(5)).cpu().numpy()
            action=int(np.argmax([self.score(u[a],a,self.irreducible_error) for a in range(5)]))
        self.counts[action]+=1
        return action
