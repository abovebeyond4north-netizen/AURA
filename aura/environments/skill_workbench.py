"""Observable two-stage manipulation microbenchmark, with *hidden* action mapping.

Purpose: isolate whether empirical one-step options can transfer to a novel
composition. The task has tiny support and is NOT a navigation or AGI benchmark.
Only RGB observations, reward, and termination reach the learner; the action
mapping is private to the simulator and never returned by reset/step.
"""
from __future__ import annotations
import numpy as np

class SkillWorkbench:
    ACTIONS = 5
    def __init__(self, task='first', mechanism_seed=71, max_steps=2, observation_noise=0.0):
        if task not in ('first','second','composition'):
            raise ValueError('Invalid workbench task')
        if not (0 <= observation_noise <= .1):
            raise ValueError('Observation noise must be in [0, .1]')
        self.task, self.max_steps, self.observation_noise = task, max_steps, observation_noise
        self._mapping = tuple(map(int, np.random.default_rng(mechanism_seed).choice(self.ACTIONS,size=2,replace=False)))
        self._rng = np.random.default_rng()
        self.stage=0; self.steps=0; self.done=False

    def reset(self, *, seed=None):
        self._rng=np.random.default_rng(seed)
        self.steps=0; self.done=False
        self.stage = 1 if self.task=='second' else 0
        return self._observe(), {}

    def _observe(self):
        image=np.zeros((3,7,7),dtype=np.uint8)
        if not self.done:
            # Two RGB visual signatures. Agents are not given symbolic stage IDs.
            image[:,2:5,2:5] = (np.array([230,35,55]) if self.stage==0 else np.array([35,230,75]))[:,None,None]
        if self.observation_noise:
            jitter=self._rng.normal(0,255*self.observation_noise,size=image.shape)
            image=np.clip(image.astype(float)+jitter,0,255).astype(np.uint8)
        return image

    def step(self, action):
        if type(action) not in (int,np.int64,np.int32) or not 0<=action<self.ACTIONS:
            raise ValueError('Invalid action')
        if self.done: raise RuntimeError('Call reset after episode termination')
        self.steps+=1
        correct=(action==self._mapping[self.stage])
        if correct:
            if self.task=='composition' and self.stage==0:
                self.stage=1
            else:
                self.done=True
        terminated=self.done
        truncated=(self.steps>=self.max_steps and not terminated)
        # Composite has a sparse final reward; no intermediate oracle hint.
        reward=1.0 if terminated else 0.0
        if truncated: self.done=True
        return self._observe(),reward,terminated,truncated,{}
