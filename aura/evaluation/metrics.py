"""Quality and resource metrics; no interpolated or invented benchmark data."""
import numpy as np

def learning_curve_auc(values):
    if not values: return float('nan')
    return float(np.trapezoid(values,dx=1)/max(1,len(values)-1))

def sample_efficiency(successes,threshold=.7):
    rates=np.asarray(successes,dtype=float)
    for i in range(10,len(rates)+1):
        if rates[i-10:i].mean()>=threshold:return i
    return None
