"""Environment, training, time and candidate budgets, fail closed at exhaustion."""
import time

class Budget:
    def __init__(self,interactions=1000,updates=1000,seconds=3600,candidates=0,development_interactions=0,imitation_eval_interactions=0):
        self.limits={'interactions':interactions,'updates':updates,'candidates':candidates,'development_interactions':development_interactions,'imitation_eval_interactions':imitation_eval_interactions}
        self.used={k:0 for k in self.limits}
        self.deadline=time.monotonic()+seconds
    def can_spend(self,name,count=1):
        if not isinstance(count,int) or count<0 or name not in self.limits:
            return False
        return time.monotonic()<=self.deadline and self.used[name]+count<=self.limits[name]
    def spend(self,name,count=1):
        if not self.can_spend(name,count):
            raise RuntimeError('Resource budget exhausted')
        self.used[name]+=count
    def remaining(self): return time.monotonic()<self.deadline and self.used['interactions']<self.limits['interactions']
    def value_of_computation(self,expected_gain,expected_cost): return float(expected_gain-expected_cost)


class ValueOfComputationAllocator:
    """Online cost-adjusted estimates of competing learning operators.

    These quantities are observed marginal validated gain and real measured cost,
    not claims of globally optimal meta-learning. A confidence bonus permits a
    small amount of experimentation with uncertain strategies.
    """
    def __init__(self,operators,seed=0,bonus=0.1):
        import numpy as np
        self.operators=tuple(operators)
        self.rng=np.random.default_rng(seed)
        self.bonus=float(bonus)
        self.records={o:[] for o in self.operators}
    def update(self,operator,validated_gain,measured_cost):
        import math
        if operator not in self.records: raise ValueError('Unknown operator')
        if not math.isfinite(validated_gain) or not math.isfinite(measured_cost) or measured_cost<0:
            raise ValueError('Invalid gain or cost')
        self.records[operator].append(float(validated_gain-measured_cost))
    def select(self,allowed=None):
        import math
        choices=[o for o in self.operators if allowed is None or o in allowed]
        if not choices: return None
        untried=[o for o in choices if not self.records[o]]
        if untried: return untried[0]
        def score(o):
            v=self.records[o]; mean=sum(v)/len(v)
            return mean+self.bonus*math.sqrt(2*math.log(sum(map(len,self.records.values()))+1)/len(v))
        best=max(choices,key=score)
        return best if score(best)>0 else None
