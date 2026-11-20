"""Version lineage persists even when a candidate is rejected."""
from dataclasses import asdict
import json
from pathlib import Path

class CandidateArchive:
    def __init__(self,path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.versions={'v0'}
        if self.path.exists():
            for row in self.records(): self.versions.add(row['version'])
    def records(self):
        if not self.path.exists(): return []
        with self.path.open() as f: return [json.loads(line) for line in f if line.strip()]
    def record(self,version,parent,modification,report):
        if version in self.versions: raise ValueError('Candidate version already exists')
        if parent not in self.versions: raise ValueError('Unknown parent version')
        item={'version':version,'parent':parent,'modification':asdict(modification),
              'evaluation':asdict(report),'status':'PROMOTED' if report.accepted else 'ARCHIVED'}
        with self.path.open('a') as f: f.write(json.dumps(item,sort_keys=True,allow_nan=False)+'\n')
        self.versions.add(version)
        return item
