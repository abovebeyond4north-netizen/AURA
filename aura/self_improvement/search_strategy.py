"""Adapt proposal-operator selection using validated feedback, not test labels."""
import numpy as np
import math
from numbers import Real

class ImprovementSearch:
    def __init__(self,seed=0):
        self.rng=np.random.default_rng(seed)
        self.operations=['increase_lr','decrease_lr','increase_capacity','decrease_exploration']
        self.counts={x:0 for x in self.operations}; self.scores={x:0. for x in self.operations}
    def select(self):
        unseen=[op for op in self.operations if self.counts[op]==0]
        if unseen: return unseen[0]
        n=sum(self.counts.values())
        return max(self.operations,key=lambda x:self.scores[x]/self.counts[x]+np.sqrt(2*np.log(n+1)/self.counts[x]))
    def update(self,operator,validated_gain,evaluation_cost):
        if operator not in self.counts: raise PermissionError('Unknown search operator')
        def finite_real(value):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                return False
            try:
                return math.isfinite(value)
            except OverflowError:
                return False
        if not finite_real(validated_gain) or not finite_real(evaluation_cost) or evaluation_cost < 0:
            raise ValueError('Finite gain and nonnegative finite evaluation cost required')
        score = self.scores[operator] + float(validated_gain) - .001*float(evaluation_cost)
        if not math.isfinite(score):
            raise ValueError('Feedback score overflow')
        self.counts[operator] += 1
        self.scores[operator] = score
