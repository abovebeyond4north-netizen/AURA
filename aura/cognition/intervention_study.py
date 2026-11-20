"""Budget-limited sequential arm balancing and observational-vs-causal audit."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from aura.cognition.causal_reasoning import compare_randomized_interventions
from aura.environments.causal_workbench import decode_visual_outcome

@dataclass
class CausalStudy:
    treatment: list[int]
    outcome: list[int]
    evidence_ids: list[str]
    assignment: str

class BalancedInterventionSelector:
    def __init__(self,seed=0):
        self.rng=np.random.default_rng(seed)
        self.n=[0,0]
    def select(self):
        if self.n[0]==self.n[1]: action=int(self.rng.integers(2))
        else: action=int(self.n[1]<self.n[0])
        self.n[action]+=1
        return action

def run_causal_study(world, samples=240, seed=0, randomized=True, equiv_margin=.2):
    if samples<40: raise ValueError('Minimum 40 trials')
    selector=BalancedInterventionSelector(seed)
    exposures=[]; outcomes=[]; ids=[]
    for idx in range(samples):
        proposed_action=selector.select() if randomized else None
        result=world.sample(intervention=proposed_action)
        observed=decode_visual_outcome(result['observation'])
        exposures.append(result['treatment']);outcomes.append(observed)
        ids.append(f'{"do" if randomized else "obs"}:{seed}:{idx}')
    study=CausalStudy(exposures,outcomes,ids,
                      'RANDOMIZED_INTERVENTION' if randomized else 'OBSERVATIONAL')
    # Never label an observational estimate as a causal effect.
    if randomized:
        estimate=compare_randomized_interventions(study.treatment,study.outcome,minimum_per_arm=20)
    else:
        a=np.asarray(exposures,dtype=bool); y=np.asarray(outcomes,dtype=float)
        estimate={'identified':False,'association':float(y[a].mean()-y[~a].mean()),
                  'reason':'Passive assignment is confounded'}
    if not estimate['identified']:
        classification='NOT_IDENTIFIED'
    else:
        low,high=estimate['ci95']
        if low>equiv_margin: classification='POSITIVE_EFFECT_SUPPORTED'
        elif high<-equiv_margin: classification='NEGATIVE_EFFECT_SUPPORTED'
        elif -equiv_margin < low and high < equiv_margin: classification='NEGLIGIBLE_EFFECT_SUPPORTED'
        else: classification='INCONCLUSIVE'
    return {'assignment':study.assignment,'n':samples,
            'result':classification,'estimate':estimate,'evidence_count':len(ids),
            'assumptions':['randomization' if randomized else 'passive correlation',
                           'stable mechanism', 'independent trials', 'reliable RGB sensor']}
