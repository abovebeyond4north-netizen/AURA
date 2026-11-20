"""Experimental reward-grounded route reuse over unlabeled image observations.

A positive terminal reward allows a real action trajectory to become a candidate.
The learner may retry it from a matching start view, checking each subsequent
view before continuing. A single rewarded trajectory is NOT a verified skill.
Static layouts and visible-state aliasing are important limits.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from aura.exploration.visual_frontier import VisualFrontierExplorer


@dataclass
class RewardRoute:
    observations: tuple
    actions: tuple
    evidence_ids: tuple
    reproductions: int = 0
    attempts: int = 0


class RewardRouteExplorer:
    def __init__(self, seed=0, *, max_routes=8, max_route_steps=256,
                 similarity=.65, retry_probability=.85):
        if not 1<=max_routes<=100 or not 1<=max_route_steps<=1000 or not 0<similarity<=1 or not 0<=retry_probability<=1:
            raise ValueError('Invalid route bounds')
        self.frontier=VisualFrontierExplorer(seed=seed)
        self.rng=np.random.default_rng(seed+18301)
        self.max_routes=max_routes; self.max_route_steps=max_route_steps
        self.similarity=similarity; self.retry_probability=retry_probability
        self.routes=[]; self.discovered=0; self.reproduced=0; self.aborted=0
        self.total_interactions=0; self._episode_index=0
        self._clear_episode()

    def _clear_episode(self):
        self.observations=[]; self.actions=[]; self.evidence=[]
        self.active=None; self.position=0; self.replay_attempt=None; self.overflow=False
        self.pending=None; self.finished=False

    def reset_episode(self):
        self.frontier.reset_episode(); self._episode_index+=1; self._clear_episode()

    @staticmethod
    def match(a,b):
        return float(np.mean(np.all(a==b,axis=0)))

    def choose(self, observation):
        obs=np.asarray(observation,dtype=np.uint8)
        if obs.shape!=(3,7,7) or self.pending is not None or self.finished:
            raise ValueError('Expected a new observation in an active episode')
        if not self.actions and self.routes and self.rng.random()<self.retry_probability:
            eligible=[r for r in self.routes if self.match(obs,r.observations[0])>=self.similarity]
            if eligible:
                # Prefer empirically reproducible routes, then shorter routes.
                self.active=max(eligible,key=lambda r:((r.reproductions+1)/(r.attempts+2),-len(r.actions)))
                self.active.attempts+=1; self.replay_attempt=self.active
        if self.active is not None:
            if (self.position<len(self.active.actions) and
                    self.match(obs,self.active.observations[self.position])>=self.similarity):
                action=self.active.actions[self.position]; self.position+=1
                self.frontier.track_action(obs,action)
            else:
                self.active=None; self.aborted+=1
                action=self.frontier.choose(obs)
        else:
            action=self.frontier.choose(obs)
        self.pending=(obs.copy(),action)
        return action

    def observe(self, observation, action, next_observation, *, reward, terminated, truncated, evidence_id=None):
        if self.pending is None or self.pending[1]!=action or not np.array_equal(self.pending[0],observation):
            raise ValueError('Observed transition does not match selected action')
        if not np.isfinite(reward): raise ValueError('Nonfinite reward')
        self.frontier.observe(observation,action,next_observation)
        self.pending=None; self.total_interactions+=1
        if len(self.actions)<self.max_route_steps:
            self.observations.append(np.asarray(observation,dtype=np.uint8).copy())
            self.actions.append(int(action)); self.evidence.append(str(evidence_id or f'{self._episode_index}:{len(self.actions)}'))
        else:self.overflow=True
        if terminated or truncated:
            self.finished=True
            if terminated and reward>0 and not self.overflow:
                actions=tuple(self.actions)
                if (self.replay_attempt is not None and self.active is self.replay_attempt and
                        actions==self.replay_attempt.actions and self.position==len(actions)):
                    self.replay_attempt.reproductions+=1; self.reproduced+=1
                elif not any(r.actions==actions and self.match(r.observations[0],self.observations[0])>=self.similarity for r in self.routes):
                    self.routes.append(RewardRoute(tuple(self.observations),actions,tuple(self.evidence)))
                    self.discovered+=1
                    if len(self.routes)>self.max_routes:
                        self.routes.sort(key=lambda r:((r.reproductions+1)/(r.attempts+2),-len(r.actions)),reverse=True)
                        self.routes.pop()
        return None

    def metrics(self):
        return {'interactions':self.total_interactions,'candidate_routes_discovered':self.discovered,
                'exact_successful_reproductions':self.reproduced,'aborted_replays':self.aborted,
                'retained_routes': [ {'length':len(r.actions),'attempts':r.attempts,
                                      'reproductions':r.reproductions,'evidence_ids':list(r.evidence_ids)} for r in self.routes],
                'scope':'positive-terminal reward route candidates; not independent skill verification'}
