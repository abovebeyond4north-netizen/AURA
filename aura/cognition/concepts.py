"""Compact clustering with predictive-gain complexity gating; not semantic labels."""
from __future__ import annotations
import numpy as np

class ConceptDiscovery:
    def __init__(self,max_clusters=8,penalty=0.01):
        self.centers=None; self.max_clusters=max_clusters; self.penalty=penalty
        self.history=[]
    def fit(self,embeddings, targets, validation=None):
        x=np.asarray(embeddings,dtype=float); y=np.asarray(targets,dtype=float)
        if len(x)<8: return {'accepted':False,'reason':'insufficient samples'}
        if validation is None: cut=max(4,int(len(x)*.7)); xt,xv=x[:cut],x[cut:]; yt,yv=y[:cut],y[cut:]
        else: xt,yt,xv,yv=x,y,np.asarray(validation[0]),np.asarray(validation[1])
        if len(xv)==0: return {'accepted':False,'reason':'no heldout examples'}
        base=float(((yv-yt.mean(axis=0))**2).mean())
        best=(base,None,None)
        for k in range(2,min(self.max_clusters,len(xt)//3)+1):
            rng=np.random.default_rng(124+k)
            centers=xt[rng.choice(len(xt),k,replace=False)].copy()
            for _ in range(15):
                ids=((xt[:,None]-centers[None])**2).sum(-1).argmin(-1)
                centers=np.stack([xt[ids==i].mean(0) if (ids==i).any() else centers[i] for i in range(k)])
            ids=((xt[:,None]-centers[None])**2).sum(-1).argmin(-1)
            pred=np.stack([yt[ids==i].mean(0) if (ids==i).any() else yt.mean(0) for i in range(k)])
            val_ids=((xv[:,None]-centers[None])**2).sum(-1).argmin(-1)
            loss=float(((yv-pred[val_ids])**2).mean())+self.penalty*k
            if loss<best[0]: best=(loss,centers,pred)
        if best[1] is None: return {'accepted':False,'reason':'no validated gain','baseline':base}
        self.centers=best[1]
        result={'accepted':True,'baseline':base,'score':best[0], 'clusters':len(self.centers)}
        self.history.append(result)
        return result
    def assign(self,x):
        if self.centers is None: raise ValueError('No accepted concepts')
        x=np.asarray(x); return ((x[:,None]-self.centers[None])**2).sum(-1).argmin(-1)
