"""Paired normal-approx and bootstrap bounds. No promotion without evidence."""
import numpy as np

def paired_bootstrap(differences,n_boot=2000,seed=91):
    x=np.asarray(differences,dtype=float)
    if len(x)<3 or not np.isfinite(x).all(): return (-1e100,1e100)
    rng=np.random.default_rng(seed)
    estimates=np.mean(x[rng.integers(0,len(x),size=(n_boot,len(x)))],axis=1)
    return tuple(map(float,np.quantile(estimates,[.025,.975])))
