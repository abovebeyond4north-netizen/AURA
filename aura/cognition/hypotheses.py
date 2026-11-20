"""Beta-Bernoulli effect hypotheses, with explicit inconclusive and all-wrong states."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from aura.types import Hypothesis

class HypothesisEngine:
    def __init__(self, min_effect=0.12):
        self.counts=defaultdict(lambda:[1.,1.])
        self.refs=defaultdict(lambda:[[],[]])
        self.min_effect=min_effect
        self.hypotheses={}
    def investigate_door(self, episode_factory, ledger, **design):
        """Run the active gridworld policy and retain its evidence-linked hypotheses."""
        from aura.cognition.grid_door_study import run
        return run(self, episode_factory, ledger, **design)

    def update(self, context, action, effect_observed: bool, evidence_id, intervened=True):
        key=(str(context),int(action),bool(intervened))
        ab=self.counts[key]
        ab[0 if effect_observed else 1]+=1
        self.refs[key][0 if effect_observed else 1].append(evidence_id)
        return self.form(context,action,intervened)
    def form(self,context,action,intervened=True):
        key=(str(context),int(action),bool(intervened))
        a,b=self.counts[key]
        prob=a/(a+b)
        h=Hypothesis(hypothesis_id=f'effect:{context}:{action}:{int(intervened)}',
          expression={'context':context,'action':int(action),'predicted_p':prob,'kind':'interventional' if intervened else 'observational'},
          assumptions=['Effect is detectable from available observations', 'Trials share a mechanism'],
          evidence_ids=list(self.refs[key][0]),counterevidence_ids=list(self.refs[key][1]),
          confidence=float(max(prob,1-prob)),applicable_contexts=[str(context)],
          proposed_tests=[{'action':int(action),'repeat':3}])
        self.hypotheses[key]=h
        return h
    def classify(self,context,action,intervened=True):
        key=(str(context),int(action),bool(intervened))
        a,b=self.counts[key]
        n=a+b-2
        if n<5: return 'INCONCLUSIVE'
        se=np.sqrt((a*b)/((a+b)**2*(a+b+1)))
        p=a/(a+b)
        if p+1.96*se<.5-self.min_effect: return 'EFFECT_UNLIKELY'
        if p-1.96*se>.5+self.min_effect: return 'EFFECT_LIKELY'
        return 'INCONCLUSIVE_OR_MISSPECIFIED'
