"""Append hypothesis revisions; superseded records remain inspectable."""
from dataclasses import asdict
from aura.types import Hypothesis, Source

class SemanticMemory:
    def __init__(self,ledger):
        self.ledger=ledger
        self.version={}
        for record in ledger.records('hypothesis'):
            stem,sep,version=record['record_id'].rpartition(':v')
            if sep and version.isdigit(): self.version[stem]=max(int(version),self.version.get(stem,0))

    def revise(self,hypothesis: Hypothesis):
        n=self.version.get(hypothesis.hypothesis_id,0)+1
        self.ledger.append(f'{hypothesis.hypothesis_id}:v{n}','hypothesis',Source.INFERRED,asdict(hypothesis))
        self.version[hypothesis.hypothesis_id]=n
