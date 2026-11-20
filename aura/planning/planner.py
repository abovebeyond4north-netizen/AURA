"""Bounded one-step curiosity/value selection; shorten horizon under uncertainty."""
import numpy as np
class Planner:
    def __init__(self,world,actor,curiosity,uncertainty_limit=.2,uncertainty_estimator=None):
        self.world,self.actor,self.curiosity=world,actor,curiosity
        self.uncertainty_limit=uncertainty_limit
        self.uncertainty_estimator=uncertainty_estimator
    def choose(self,obs,rng,horizon=4):
        u=float(self.world.estimate_uncertainty(np.repeat(obs[None],5,axis=0),np.arange(5)).mean())
        if (u>self.uncertainty_limit or
                (self.uncertainty_estimator is not None and
                 self.uncertainty_estimator.detect_distribution_shift(obs))):
            return self.curiosity.choose(obs,rng=rng),1
        action=self.actor.select(self.world,obs,.15,rng)
        return action,min(horizon,4)
