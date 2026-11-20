"""Only an intervention/randomization distinguishes observation from causation."""
import numpy as np

def compare_randomized_interventions(treatments, outcomes, minimum_per_arm=5):
    """Return estimated average treatment effect with a normal-approx CI.
    Caller must attest treatment assignment was randomized; otherwise estimate
    is an association, not a causal effect.
    """
    a=np.asarray(treatments,dtype=bool); y=np.asarray(outcomes,dtype=float)
    if len(a)!=len(y) or len(y)<2*minimum_per_arm: return {'identified':False,'reason':'insufficient evidence'}
    pos,neg=y[a],y[~a]
    if min(len(pos),len(neg))<minimum_per_arm: return {'identified':False,'reason':'unbalanced interventions'}
    effect=float(pos.mean()-neg.mean())
    se=float(np.sqrt(pos.var(ddof=1)/len(pos)+neg.var(ddof=1)/len(neg)))
    return {'identified':True,'effect':effect,'ci95':(effect-1.96*se,effect+1.96*se),
            'assumption':'Randomized treatment, consistency, and no cross-trial interference required'}
