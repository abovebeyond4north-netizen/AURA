"""Visual causal intervention benchmark with an unobserved common cause.

When passive, treatment follows a hidden confounder; interventions randomize
that treatment and reveal whether it changes the outcome. Only observed RGB
outcomes, randomized assignments and externally specified study design are
provided to the experimental learner.
"""
from __future__ import annotations
import numpy as np

class CausalWorkbench:
    def __init__(self, mechanism='confounded', seed=0):
        if mechanism not in ('confounded','effective'):
            raise ValueError('Invalid mechanism')
        self._mechanism=mechanism
        self._rng=np.random.default_rng(seed)

    def sample(self, intervention=None):
        if intervention not in (None,0,1): raise ValueError('Intervention must be 0, 1 or None')
        confounder=int(self._rng.integers(2))
        naturally_assigned = int(self._rng.random() < (.9 if confounder else .1))
        treatment = naturally_assigned if intervention is None else int(intervention)
        if self._mechanism=='confounded':
            result=confounder
        else:
            result=treatment if self._rng.random()<.85 else confounder
        image=np.zeros((3,7,7),dtype=np.uint8)
        image[:,2:5,2:5] = (np.array([12,240,10]) if result else np.array([240,10,12]))[:,None,None]
        return {'observation':image,'treatment':treatment,
                'assignment':'OBSERVATIONAL' if intervention is None else 'RANDOMIZED_INTERVENTION'}

def decode_visual_outcome(image):
    """A predeclared, sensor-level RGB decoder, not a simulator mechanism oracle."""
    arr=np.asarray(image)
    if arr.shape!=(3,7,7): raise ValueError('Wrong image shape')
    return int(arr[1,3,3] > arr[0,3,3])
