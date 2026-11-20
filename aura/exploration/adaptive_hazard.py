"""Online fractional reset-rate estimator, conditional on certified recoveries."""
from math import fsum
from aura.exploration.hazard_dynamics import HazardDynamicsCamera
from aura.exploration.learned_dynamics import PERMUTATIONS

class ResetRate:
    def __init__(self, alpha=0.6, beta=19.4):
        if not alpha > 0 or not beta > 0:
            raise ValueError('positive prior required')
        self.alpha, self.beta = alpha, beta
        self.updates = 0

    @property
    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    def observe(self, prior, posterior, hazard):
        # Probability of uniform reset conditional on each resulting mapping.
        # A reset may reproduce the original mapping.
        reset = hazard / len(PERMUTATIONS)
        mass = fsum(posterior.get(m, 0) * reset /
                    ((1-hazard)*prior.get(m, 0)+reset) for m in PERMUTATIONS)
        self.alpha += mass
        self.beta += 1-mass
        self.updates += 1

class AdaptiveHazardCamera(HazardDynamicsCamera):
    def __init__(self, image, step):
        self.rate = ResetRate()
        super().__init__(image, step)

    @property
    def hazard(self):
        return self.rate.mean

    def _marginal(self, weights):
        prior = self.mapping_belief or {self.mapping: 1.}
        hazard = self.hazard
        super()._marginal(weights)
        self.rate.observe(prior, self.mapping_belief, hazard)
