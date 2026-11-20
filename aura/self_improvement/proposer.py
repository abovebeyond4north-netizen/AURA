"""Constrained proposals based on measured plateau or instability."""
from aura.types import Modification
import math

class ModificationProposer:
    def __init__(self, search): self.search=search
    def propose(self,version,weakness):
        try:
            valid = type(weakness) in (int, float) and math.isfinite(weakness) and weakness >= 0
        except OverflowError:
            valid = False
        if not valid: raise ValueError('Measured finite nonnegative weakness required')
        name=self.search.select()
        variants={
          'increase_lr':('learning_rate','multiply',1.5),
          'decrease_lr':('learning_rate','multiply',0.7),
          'increase_capacity':('model_capacity','set',64),
          'decrease_exploration':('exploration_rate','set',0.08),
        }
        t,operation,v=variants[name]
        return name, Modification(t,operation,{'value':v},version)
