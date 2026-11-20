"""Action-conditional self-supervised curiosity with a frozen local validation bank.

This is a compact, non-neural comparator for studying exploration in the
procedural *image* world. It learns action-conditioned residual dynamics from
fixed, unlabeled image statistics, and measures learning progress on samples
that are NEVER used for parameter updates.

The reference bank is chosen online from early experiences; it is not a
protected benchmark. An error floor is only a proxy for aleatoric noise and can
also capture unmodelled hidden states, covariate shift or insufficient capacity.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import numpy as np
from aura.world_model.uncertainty import UncertaintyEstimator


@dataclass(frozen=True)
class ExplorationUpdate:
    action: int
    role: str
    before_loss: float | None
    after_loss: float | None
    validated_gain: float


class ValidatedProgressExplorer:
    """Bounded five-action online model; never accesses simulator diagnostics.

    Parameters are randomly initialized without pretrained weights.
    Predictor is x_next = x + [x, 1] W[action]. The first K transitions per
    action become immutable validation pairs; only subsequent pairs update W.
    """
    def __init__(self, seed=0, n_actions=5, validation_per_action=4,
                 lr=.3, epsilon=.08, max_state_visits=10000):
        if n_actions < 2 or validation_per_action < 2 or lr <= 0 or not 0 <= epsilon <= 1:
            raise ValueError('Invalid explorer configuration')
        self.rng=np.random.default_rng(seed)
        self.n_actions=n_actions
        self.validation_per_action=validation_per_action
        self.lr=lr
        self.epsilon=epsilon
        self.feature_dim=12
        self.weights=self.rng.normal(0,.001,(n_actions,self.feature_dim+1,self.feature_dim))
        self.validation=[[] for _ in range(n_actions)]
        self.training_count=np.zeros(n_actions,dtype=int)
        self.action_count=np.zeros(n_actions,dtype=int)
        self.progress=np.zeros(n_actions,dtype=float)
        self.visits=Counter()
        self.max_state_visits=max_state_visits
        self.update_log=[]
        self.palette=Counter()
        self.predictable_change=np.zeros(n_actions,dtype=float)
        self.effect_observations=np.zeros(n_actions,dtype=int)

    @staticmethod
    def features(image):
        result=UncertaintyEstimator._features(image)
        if not np.all(np.isfinite(result)):
            raise ValueError('Non-finite image features')
        return result

    @staticmethod
    def state_key(features):
        # Coarse, generic visual statistics; no goal/object labels or tile colors.
        return tuple(np.clip(np.rint(features/.12),0,8).astype('int8').tolist())

    @staticmethod
    def _pixel_codes(image):
        arr=np.asarray(image,dtype=np.uint8)
        if arr.shape!=(3,7,7):raise ValueError('Expected CHW 3x7x7 RGB image')
        r,g,b=(arr[0].astype(np.uint32),arr[1].astype(np.uint32),arr[2].astype(np.uint32))
        return (r<<16)|(g<<8)|b

    def _stable_effect(self,observation,next_observation):
        """Effect on recurring colors, excluding single-frame random pixels.

        Palette is learned through counted observations rather than any
        simulator color table. This is a generic but limited inductive prior.
        """
        before=self._pixel_codes(observation)
        after=self._pixel_codes(next_observation)
        # Determine rarity using previous observations to avoid self-labeling
        # a random pixel as stable in the observation that introduces it.
        recurring={k for k,v in self.palette.items() if v>=4}
        if recurring:
            eligible=np.isin(before,list(recurring)) & np.isin(after,list(recurring))
            effect=float(np.mean(eligible & (before!=after)))
        else:effect=0.
        # Update only on real, observed pixel arrays, never imagined samples.
        self.palette.update(map(int,before.reshape(-1)))
        self.palette.update(map(int,after.reshape(-1)))
        if len(self.palette)>8000:
            self.palette=Counter(dict(self.palette.most_common(4000)))
        return effect

    def _context_key(self,observation):
        """Unsupervised recurrent-color map, masked for one-off visual noise."""
        if sum(self.palette.values())<400:
            return self.state_key(self.features(observation))
        common={k for k,v in self.palette.items() if v>=8}
        codes=self._pixel_codes(observation)
        return tuple(np.where(np.isin(codes,list(common)),codes,-1).reshape(-1).tolist())

    def prediction(self,action,x):
        return x+np.concatenate((x,[1.])) @ self.weights[action]

    def validation_loss(self,action):
        pairs=self.validation[action]
        if not pairs:return None
        return float(np.mean([np.square(self.prediction(action,x)-y).mean() for x,y in pairs]))

    def scores(self,observation,policy='validated_progress'):
        if policy not in ('validated_progress','prediction_error'):
            raise ValueError('Unknown exploration policy')
        x=self.features(observation)
        context=self._context_key(observation)
        scores=[]
        for a in range(self.n_actions):
            loss=self.validation_loss(a)
            if policy=='prediction_error':
                # Explicit counterfactual: raw surprise prefers noisy action.
                value=float(loss if loss is not None else .1)
            else:
                # Learning progress is action-specific and measured on FIXED
                # examples. The residual penalty suppresses persistent error
                # floors without incorrectly claiming an aleatoric estimate.
                floor=float(loss if loss is not None else 0.)
                novelty=.10/np.sqrt(1+self.action_count[a])
                context_novelty=.25/np.sqrt(1+self.visits[(context,a)])
                # A gain must generalize to independent validation halves.
                # Cap its weight: high, possibly noisy error must not cause
                # perpetual selection after learning stalls.
                learning=max(0.,4.*self.progress[a]/(floor+.002))
                visual_effect=(self.predictable_change[a]
                               if self.effect_observations[a]>=4 else 0.)
                value=(novelty+context_novelty+min(.05,learning)
                       +.9*visual_effect-10.*floor)
            scores.append(value)
        return np.asarray(scores,dtype=float)

    def choose(self,observation,policy='validated_progress',epsilon=None,rng=None):
        chooser=self.rng if rng is None else rng
        explore=self.epsilon if epsilon is None else epsilon
        if not 0<=explore<=1:raise ValueError('epsilon must be in [0,1]')
        if policy=='random' or chooser.random()<explore:
            action=int(chooser.integers(self.n_actions))
        else:
            scores=self.scores(observation,policy)
            # Random tie-breaking prevents a stable bias toward action zero.
            action=int(chooser.choice(np.flatnonzero(np.isclose(scores,scores.max()))))
        context=self._context_key(observation)
        if self.visits[(context,action)] < self.max_state_visits:
            self.visits[(context,action)]+=1
        self.action_count[action]+=1
        return action

    def observe(self,observation,action,next_observation):
        if not isinstance(action,(int,np.integer)) or not 0<=int(action)<self.n_actions:
            raise ValueError('Invalid action')
        x=self.features(observation);y=self.features(next_observation)
        effect=self._stable_effect(observation,next_observation)
        n=self.effect_observations[action]
        self.predictable_change[action]=(n*self.predictable_change[action]+effect)/(n+1)
        self.effect_observations[action]+=1
        if len(self.validation[action])<self.validation_per_action:
            self.validation[action].append((x.copy(),y.copy()))
            result=ExplorationUpdate(int(action),'validation',None,None,0.)
        else:
            before=self.validation_loss(action)
            heldout=self.validation[action]
            half=len(heldout)//2
            def half_loss(subset):
                return float(np.mean([np.square(self.prediction(action,v)-t).mean()
                                      for v,t in subset]))
            before_halves=[half_loss(heldout[:half]),half_loss(heldout[half:])]
            aug=np.r_[x,1.]
            error=self.prediction(action,x)-y
            # Gradient of mean-squared feature prediction, safely clipped.
            grad=np.outer(aug,error)*2./self.feature_dim
            np.clip(grad,-.5,.5,out=grad)
            self.weights[action]-=self.lr*grad
            self.training_count[action]+=1
            after=self.validation_loss(action)
            after_halves=[half_loss(heldout[:half]),half_loss(heldout[half:])]
            # Conservative signed improvement. Upward fluctuations in one
            # held-out half are not mistaken for positive information gain.
            gain=min(b-a for b,a in zip(before_halves,after_halves))
            self.progress[action]=.90*self.progress[action]+.10*gain
            result=ExplorationUpdate(int(action),'training',before,after,gain)
        self.update_log.append(result)
        return result

    def metrics(self):
        return {'validation_examples_by_action':[len(b) for b in self.validation],
                'training_updates_by_action':self.training_count.tolist(),
                'action_counts':self.action_count.tolist(),
                'heldout_mse_by_action':[self.validation_loss(a) for a in range(self.n_actions)],
                'validated_progress_by_action':self.progress.tolist(),
                'recurring_color_change_by_action':self.predictable_change.tolist(),
                'method':'frozen online validation; progress is an observed local proxy'}
