"""Bounded factual replay; imagined transitions cannot enter this collection."""
from collections import deque
import random
import numpy as np
from aura.types import Experience, Source

class ExperienceReplay:
    def __init__(self, capacity=10000): self.data=deque(maxlen=capacity)
    def add(self, experience: Experience):
        if experience.source != Source.OBSERVED: raise ValueError('Replay only accepts observed experience')
        self.data.append(experience)
    def __len__(self): return len(self.data)
    def sample(self,n):
        if len(self.data)<n: raise ValueError('Insufficient replay data')
        items=random.sample(list(self.data),n)
        return (np.stack([x.observation for x in items]),
                np.asarray([x.action for x in items],dtype=np.int64),
                np.stack([x.next_observation for x in items]))
