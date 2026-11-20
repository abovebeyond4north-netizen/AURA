"""Select interventions that separate structured competing hypotheses."""
import numpy as np

def binary_entropy(prob):
    p=np.clip(prob,1e-7,1-1e-7)
    return -p*np.log(p)-(1-p)*np.log(1-p)

class ExperimentSelector:
    @staticmethod
    def choose(hypotheses, actions=(0,1,2,3,4)):
        if not hypotheses: return 4
        # Competing candidate intervention effects. Distinguish by disagreement.
        utilities=[]
        for a in actions:
            ps=[h.expression.get('predicted_p',0.5) if h.expression.get('action')==a else 0.5 for h in hypotheses]
            utilities.append(float(np.var(ps))+float(np.mean([binary_entropy(p) for p in ps]))*.01)
        return int(actions[int(np.argmax(utilities))])
