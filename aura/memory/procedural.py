"""Versioned procedural findings backed by evidence references."""
from dataclasses import asdict
from aura.types import Skill, Source

class ProceduralMemory:
    def __init__(self,ledger):
        self.ledger=ledger
        self.version={}
        for record in ledger.records('skill'):
            stem,sep,version=record['record_id'].rpartition(':v')
            if sep and version.isdigit(): self.version[stem]=max(int(version),self.version.get(stem,0))

    def revise(self,skill:Skill):
        n=self.version.get(skill.skill_id,0)+1
        self.ledger.append(f'{skill.skill_id}:v{n}','skill',Source.INFERRED,asdict(skill))
        self.version[skill.skill_id]=n
