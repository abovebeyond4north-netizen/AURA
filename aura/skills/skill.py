"""Learned empirical action sequences, not symbolic oracle policies."""
from aura.types import Skill

class SequenceSkill:
    def __init__(self, skill: Skill, actions: list[int]):
        if not actions or any(a not in range(5) for a in actions): raise ValueError('Invalid skill actions')
        self.skill=skill; self.actions=tuple(actions)
    def applicable(self,context): return context in self.skill.applicable_contexts
    def execute(self,env,context):
        if not self.applicable(context): raise ValueError('Skill preconditions violated')
        total=0.; done=False; truncated=False
        for action in self.actions:
            _,r,done,truncated,_=env.step(action)
            total+=r
            if done or truncated: break
        if done: self.skill.successes+=1
        else: self.skill.failures+=1
        return total,done,truncated
