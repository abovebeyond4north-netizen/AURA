"""Unlabeled visual prototypes and reward-driven empirical option discovery.

No task name, simulator mechanism, diagnostic state or action key is passed to
this learner. Observations are compressed by unlabeled prototype matching. An
option is accepted only when repeated positive-reward observations support it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class VisualPrototype:
    centroid: np.ndarray
    count: int = 1

class VisualPrototypeEncoder:
    def __init__(self, threshold=0.12):
        self.threshold=float(threshold)
        self.prototypes: list[VisualPrototype]=[]

    @staticmethod
    def features(observation):
        x=np.asarray(observation,dtype=np.float32)
        if x.shape!=(3,7,7): raise ValueError('Expected 3x7x7 RGB observation')
        return x[:,2:5,2:5].mean(axis=(1,2))/255.

    def encode(self, observation, learn=True):
        x=self.features(observation)
        if self.prototypes:
            distances=[np.linalg.norm(p.centroid-x) for p in self.prototypes]
            closest=int(np.argmin(distances))
            if distances[closest]<=self.threshold:
                p=self.prototypes[closest]
                if learn:
                    p.count+=1
                    p.centroid+=(x-p.centroid)/p.count
                return closest
        if not learn: return None
        self.prototypes.append(VisualPrototype(x.copy()))
        return len(self.prototypes)-1

@dataclass
class LearnedOption:
    initiation_prototype: int
    action: int
    observed_positive_outcomes: int
    observed_failures: int
    evidence_ids: list[str]=field(default_factory=list)
    @property
    def empirical_success(self):
        return self.observed_positive_outcomes/max(1,self.observed_positive_outcomes+self.observed_failures)

class OnlineOptionLearner:
    def __init__(self,seed=0,n_actions=5):
        self.encoder=VisualPrototypeEncoder()
        self.rng=np.random.default_rng(seed)
        self.n_actions=n_actions
        self.outcomes: dict[tuple[int,int],list[int]]={}
        self.evidence: dict[tuple[int,int],list[str]]={}
        self.observed_transitions=0

    def choose(self,observation,epsilon=0.0):
        sid=self.encoder.encode(observation,learn=False)
        if sid is None or self.rng.random()<epsilon:
            return int(self.rng.integers(self.n_actions))
        # Untried actions are conservatively scored by the Beta(1,1) prior.
        scores=[]
        for action in range(self.n_actions):
            success,fail=self.outcomes.get((sid,action),(0,0))
            scores.append((success+1)/(success+fail+2))
        return int(np.argmax(scores))

    def observe(self,observation,action,reward,evidence_id):
        if not 0<=action<self.n_actions: raise ValueError('Invalid action')
        sid=self.encoder.encode(observation)
        entry=self.outcomes.setdefault((sid,int(action)),[0,0])
        positive=reward>0
        entry[0 if positive else 1]+=1
        self.evidence.setdefault((sid,int(action)),[]).append(str(evidence_id))
        self.observed_transitions+=1

    def options(self,min_successes=2,min_reliability=.5):
        found=[]
        for (sid,a),(good,bad) in sorted(self.outcomes.items()):
            if good>=min_successes and good/(good+bad)>=min_reliability:
                found.append(LearnedOption(sid,a,good,bad,self.evidence[(sid,a)].copy()))
        return found

    def act_with_options(self,observation,options):
        sid=self.encoder.encode(observation,learn=False)
        eligible=[o for o in options if sid==o.initiation_prototype]
        if not eligible: return self.choose(observation)
        return max(eligible,key=lambda o:(o.empirical_success,o.observed_positive_outcomes)).action

def run_skill_transfer(seed=0,train_episodes=60,test_episodes=100,observation_noise=.01):
    """Matched evaluation: learned options vs random, independent eval seeds.

    Mechanism is identical for primitive training and unseen *composition*;
    fresh observations and episodes evaluate reuse, not a new physical rule.
    """
    from aura.environments.skill_workbench import SkillWorkbench
    learner=OnlineOptionLearner(seed=seed)
    for task in ('first','second'):
        env=SkillWorkbench(task,mechanism_seed=100+seed,observation_noise=observation_noise)
        for i in range(train_episodes):
            obs,_=env.reset(seed=seed*10000+i+100000*(task=='second'))
            # Explore primitive outcomes without externally supplied labels.
            action=int(learner.rng.integers(5))
            _,r,_,_,_=env.step(action)
            learner.observe(obs,action,r,f'train-{task}-{i}')
    options=learner.options()
    scores={}
    for policy in ('options','random'):
        success=0; interactions=0
        rand=np.random.default_rng(seed+10000)
        for i in range(test_episodes):
            env=SkillWorkbench('composition',mechanism_seed=100+seed,
                               observation_noise=observation_noise)
            obs,_=env.reset(seed=1000000+seed*test_episodes+i)
            for _ in range(2):
                action=(learner.act_with_options(obs,options) if policy=='options'
                        else int(rand.integers(5)))
                obs,r,term,trunc,_=env.step(action)
                interactions+=1
                if term: success+=1
                if term or trunc: break
        scores[policy]={'wins':success,'episodes':test_episodes,'success_rate':success/test_episodes,
                        'interactions':interactions}
    return {'seed':seed,'train_episodes_per_primitive':train_episodes,
            'positive_reward_options':len(options),'visual_prototypes':len(learner.encoder.prototypes),
            'observed_training_transitions':learner.observed_transitions,
            'trained_vs_random':scores,
            'limitation':'tiny two-stage visual microbenchmark, not procedural gridworld transfer'}
