"""Empirical predictive-risk calibration and conditional variability diagnostics.

Crucial distinctions:
- ensemble disagreement: epistemic *proxy*, never a probability;
- conditional observed outcome variance: variability under repeated identical
  sensed inputs/actions, not guaranteed irreducible (latent confounding possible);
- calibrated image-MSE risk: disjoint, temporally ordered records (not an
  independent/exchangeable conformal guarantee);
- shift detector: calibrated on held-out images and frozen after fitting.
"""
from __future__ import annotations
from collections import deque, OrderedDict
from hashlib import sha256
import numpy as np


class UncertaintyEstimator:
    def __init__(self, world, threshold=.1, history=500, max_contexts=500):
        self.world,self.threshold=world,float(threshold)
        self.errors=deque(maxlen=history)
        self.predicted=deque(maxlen=history)
        self.observations=deque(maxlen=history)
        self._contexts=OrderedDict()
        self.max_contexts=int(max_contexts)
        self._shift_model=None
        if self.max_contexts<1: raise ValueError('max_contexts must be positive')

    def estimate_uncertainty(self,obs,action):
        return float(self.world.estimate_uncertainty(obs[None],np.array([action]))[0])

    def update(self,obs,action,next_obs):
        import torch
        u=self.estimate_uncertainty(obs,action)
        with torch.no_grad():
            pred=self.world.prediction(obs[None],np.array([action]))
            target=torch.as_tensor(next_obs,device=self.world.device_name,dtype=torch.float32)/255.
            error=float(((torch.stack(pred).mean(0)[0]-target)**2).mean())
        self.predicted.append(u);self.errors.append(error)
        self.observations.append(float(np.asarray(obs).mean()))
        self.record_observed_outcome(obs,action,next_obs)
        return u,error

    def record_observed_outcome(self,obs,action,next_obs):
        """Only actual outcomes; imagined predictions never enter moments."""
        before=np.asarray(obs)
        outcome=np.asarray(next_obs)
        if before.shape!=(3,7,7) or outcome.shape!=(3,7,7):
            raise ValueError('Expected 3x7x7 observed images')
        if not isinstance(action,(int,np.integer)) or not 0<=int(action)<5:
            raise ValueError('Action must be a valid discrete control')
        key=sha256(before.tobytes()+bytes([int(action)])).digest()
        y=outcome.astype(np.float64).reshape(-1)/255.
        if key in self._contexts:
            record=self._contexts.pop(key)
            record[0]+=1
            delta=y-record[1]
            record[1]+=delta/record[0]
            record[2]+=delta*(y-record[1])
        else:
            record=[1,y.copy(),np.zeros_like(y)]
        self._contexts[key]=record
        if len(self._contexts)>self.max_contexts:
            self._contexts.popitem(last=False)

    def conditional_outcome_variance(self,obs,action):
        """Return within-context measured image variance, or None if unidentifiable."""
        arr=np.asarray(obs)
        if arr.shape!=(3,7,7): raise ValueError('Expected 3x7x7 observation')
        if not isinstance(action,(int,np.integer)) or not 0<=int(action)<5:
            raise ValueError('Invalid action')
        key=sha256(arr.tobytes()+bytes([int(action)])).digest()
        record=self._contexts.get(key)
        if record is None or record[0]<2: return None
        return float(np.maximum(0,record[2]/(record[0]-1)).mean())

    def estimate_components(self,obs,action):
        """Useful uncertainty decomposition; no unearned causal/probability claims."""
        epistemic=self.estimate_uncertainty(obs,action)
        variability=self.conditional_outcome_variance(obs,action)
        return {'ensemble_disagreement':epistemic,
                'observed_conditional_variance':variability,
                'distribution_shift':self.detect_distribution_shift(obs),
                'prediction_error_bound':self.prediction_error_bound(),
                'qualification':('Observed variability can reflect partial observability. '
                                 'Neither component alone identifies model-capacity limits.')}

    def calibration(self,alpha=.1):
        """Risk bins and upper bounds evaluated on disjoint calibration records.

        Chronological dependence/training drift can invalidate nominal coverage;
        return empirical coverage instead of asserting a confidence guarantee.
        """
        n=len(self.errors)
        if n<10: return {'predictive_error_calibration_mae':None,'n':n}
        uncertainty=np.asarray(self.predicted,dtype=float)
        residual=np.asarray(self.errors,dtype=float)
        if not np.isfinite(uncertainty).all() or not np.isfinite(residual).all():
            raise ValueError('Non-finite risk observations')
        train=np.arange(n)%2==0
        u_train,u_test=uncertainty[train],uncertainty[~train]
        err_train,err_test=residual[train],residual[~train]
        thresholds=np.quantile(u_train,[.25,.5,.75])
        a=np.searchsorted(thresholds,u_train,side='right')
        b=np.searchsorted(thresholds,u_test,side='right')
        global_error=float(err_train.mean())
        estimates=np.array([float(err_train[a==k].mean()) if (a==k).any()
                            else global_error for k in range(4)])
        bounds=np.array([float(np.quantile(err_train[a==k],1-alpha))
                         if (a==k).any() else float(np.quantile(err_train,1-alpha))
                         for k in range(4)])
        predicted=estimates[b]
        return {'predictive_error_calibration_mae':float(np.abs(predicted-err_test).mean()),
                'empirical_error_bound_coverage':float((err_test<=bounds[b]).mean()),
                'requested_upper_bound_coverage':1-alpha,
                'n':n,'description':'disjoint record risk bins; not a probability/exchangeability guarantee'}

    def prediction_error_bound(self,alpha=.1):
        if len(self.errors)<25:return None
        return float(np.quantile(np.asarray(self.errors,dtype=float),1-alpha))

    @staticmethod
    def _features(observation):
        """Fixed, label-free image summary; no trainable encoder drift."""
        x=np.asarray(observation,dtype=float)
        if x.shape!=(3,7,7):raise ValueError('Expected 3x7x7 image')
        x=x/255.
        # Per-channel mean/std, global center and border means.
        return np.concatenate((x.mean(axis=(1,2)),x.std(axis=(1,2)),
                               x[:,2:5,2:5].mean(axis=(1,2)),
                               np.concatenate((x[:,0,:],x[:,-1,:],
                                               x[:,:,0],x[:,:,-1]),axis=1).mean(axis=1)))

    @property
    def shift_is_calibrated(self):
        return self._shift_model is not None

    def fit_shift_detector(self,reference_observations,alpha=.05):
        """Calibrate nearest-reference anomaly scores on disjoint images.

        Separate reference/calibration halves are used. No test samples are
        added after fitting, protecting the threshold from feedback leakage.
        """
        if not 0<alpha<.5:raise ValueError('alpha must be in (0,0.5)')
        x=np.asarray([self._features(o) for o in reference_observations],dtype=float)
        if len(x)<30:raise ValueError('Need at least 30 observations for shift calibration')
        ref,cal=x[::2],x[1::2]
        center=np.mean(ref,axis=0)
        scale=np.maximum(np.std(ref,axis=0),.02)
        normalized=(ref-center)/scale
        scores=np.min(np.linalg.norm((cal[:,None,:]-ref[None,:,:])/scale,axis=2),axis=1)
        # Conservative finite-sample empirical rank for reference-distance scores.
        scores=np.sort(scores)
        rank=min(len(scores)-1,max(0,int(np.ceil((len(scores)+1)*(1-alpha)))-1))
        self._shift_model=(ref.copy(),scale.copy(),float(scores[rank]))
        return {'reference_size':len(ref),'calibration_size':len(cal),
                'threshold':float(scores[rank]),'alpha':alpha}

    def shift_report(self,observation):
        x=self._features(observation)
        if self._shift_model is not None:
            ref,scale,threshold=self._shift_model
            score=float(np.min(np.linalg.norm((ref-x)/scale,axis=1)))
            return {'shift':bool(score>threshold),'score':score,
                    'threshold':threshold,'method':'frozen_empirical_nearest_reference'}
        if len(self.observations)<25:
            return {'shift':False,'score':None,'threshold':None,'method':'insufficient_evidence'}
        hist=np.asarray(self.observations)
        score=abs(float(np.asarray(observation).mean())-hist.mean())
        threshold=3*max(hist.std(),1e-3)
        return {'shift':bool(score>threshold),'score':float(score),
                'threshold':float(threshold),'method':'legacy_global_mean_heuristic'}

    def detect_distribution_shift(self,observation):
        return self.shift_report(observation)['shift']

    def estimate_prediction_confidence(self,trajectory):
        if not trajectory:return 0.
        # Risk score, never presented as a calibrated probability.
        return 1/(1+float(np.mean([x for x in trajectory])))

    def should_request_more_evidence(self,hypothesis):
        return hypothesis.confidence<.85


class TransitionRisk:
    """Frozen action-conditional residual bounds from a separate calibration set."""

    def __init__(self, calibration, alpha=.1):
        if not 0 < alpha < 1:
            raise ValueError('alpha must be between zero and one')
        self.bounds = {}
        for action, values in calibration.items():
            scores = np.asarray(values, dtype=float)
            if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
                    or (scores < 0).any()):
                raise ValueError('Expected finite nonnegative residuals')
            rank = int(np.ceil((len(scores)+1)*(1-alpha)))
            if rank <= len(scores):
                self.bounds[action] = float(np.partition(scores,rank-1)[rank-1])

    @staticmethod
    def residual(prediction, actual):
        a,b = np.asarray(prediction,dtype=float),np.asarray(actual,dtype=float)
        if a.shape != (3,7,7) or b.shape != a.shape:
            raise ValueError('Expected matching RGB frames')
        result = float(np.mean(((a-b)/255.)**2))
        if not np.isfinite(result):
            raise ValueError('Residual must be finite')
        return result

    def check(self, action, prediction, actual):
        score = self.residual(prediction,actual)
        bound = self.bounds.get(action)
        return {'error':score,'bound':bound,'shifted':None if bound is None else score>bound}
