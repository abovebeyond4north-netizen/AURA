"""Candidate concatenation of verified primitive sequences, separately retested."""
from aura.types import Skill
from .skill import SequenceSkill

def compose(first:SequenceSkill, second:SequenceSkill, context:str,max_actions=30):
    if not first.applicable(context) or not second.applicable(context): raise ValueError('Incompatible contexts')
    actions=first.actions+second.actions
    if len(actions)>max_actions: raise ValueError('Composition budget exceeded')
    skill=Skill(f'{first.skill.skill_id}+{second.skill.skill_id}',{'task_family':context},
                'compose',{'max_actions':len(actions)},
                {'effect':'unverified composition'},first.skill.evidence_ids+second.skill.evidence_ids,
                applicable_contexts=[context])
    return SequenceSkill(skill,list(actions))
