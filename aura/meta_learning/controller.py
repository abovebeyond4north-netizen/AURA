"""Budget-aware adaptive UCB selector for concrete learning operations."""
import math
import numpy as np
STRATEGIES=('EXPLORE','TRAIN_WORLD_MODEL','TEST_HYPOTHESIS','PRACTICE_SKILL',
            'REPLAY_MEMORY','REFINE_CONCEPT','ADAPT_CURRICULUM','PROPOSE_SELF_IMPROVEMENT')

class MetaController:
    def __init__(self,strategies=STRATEGIES):
        self.strategies=tuple(strategies); self.counts=np.zeros(len(strategies)); self.gains=np.zeros(len(strategies)); self.total_cost=np.zeros(len(strategies)); self.gain_counts=np.zeros(len(strategies))
    def select(self,allowed=None):
        choices=[i for i,s in enumerate(self.strategies) if allowed is None or s in allowed]
        if not choices: raise ValueError('No authorized learning strategies')
        untried=[i for i in choices if self.counts[i]==0]
        if untried: return self.strategies[untried[0]]
        total=max(1,self.counts.sum()); scores=self.gains/(self.gain_counts+1e-9)-self.total_cost/(self.counts+1e-9)
        scores+=np.sqrt(2*np.log(total+1)/(self.counts+1e-9))
        return self.strategies[max(choices,key=lambda i:scores[i])]
    def update(self,strategy,validated_gain,cost):
        idx=self.strategies.index(strategy)
        if not math.isfinite(cost) or cost<0 or (validated_gain is not None and not math.isfinite(validated_gain)):
            raise ValueError('Invalid gain or cost')
        self.counts[idx]+=1; self.total_cost[idx]+=float(cost)
        if validated_gain is not None:
            self.gains[idx]+=float(validated_gain); self.gain_counts[idx]+=1

    def metrics(self):
        return {'selections':dict(zip(self.strategies,self.counts.astype(int).tolist())),
                'measured_gains':dict(zip(self.strategies,self.gain_counts.astype(int).tolist())),
                'gain_sums':dict(zip(self.strategies,self.gains.tolist())),
                'scope':'sparse local model-update association; not causal or independent improvement'}
