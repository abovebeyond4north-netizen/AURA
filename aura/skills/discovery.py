"""Mine empirically associated sequences with distinct successful episode support."""
from collections import Counter
from aura.types import Skill
from .skill import SequenceSkill

class SkillDiscovery:
    def __init__(self, min_occurrences=2, max_length=8):
        if min_occurrences<2 or max_length<2: raise ValueError('Invalid discovery limits')
        self.counts=Counter(); self.skill_library={}
        self.min_occurrences=min_occurrences; self.max_length=max_length
        self._seen_episodes=set(); self._support={}; self._next_id=1
    def observe_episode(self,transitions,success,context):
        if not success or not transitions: return []
        episode_id=transitions[0].episode_id
        if any(x.episode_id!=episode_id for x in transitions): raise ValueError('Mixed episode evidence')
        if episode_id in self._seen_episodes: return []
        self._seen_episodes.add(episode_id)
        actions=[x.action for x in transitions]
        learned=[]; seen_sequences=set()
        for length in range(2,min(self.max_length,len(actions))+1):
            for start in range(0,len(actions)-length+1):
                seq=tuple(actions[start:start+length]); key=(str(context),seq)
                if key in seen_sequences: continue
                seen_sequences.add(key); self.counts[key]+=1
                if self.counts[key]<=self.min_occurrences:
                    self._support.setdefault(key,[]).extend(x.evidence_id for x in transitions[start:start+length])
                if self.counts[key]==self.min_occurrences:
                    refs=self._support.pop(key)
                    sid=f'seq-{self._next_id}'; self._next_id+=1
                    meta=Skill(sid,{'task_family':context},sid,{'max_actions':length},
                               {'effect':'empirically associated with distinct successful episodes'},refs,
                               applicable_contexts=[context])
                    self.skill_library[sid]=SequenceSkill(meta,list(seq)); learned.append(sid)
        return learned
