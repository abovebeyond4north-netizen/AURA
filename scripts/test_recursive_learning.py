#!/usr/bin/env python3
"""Two-stage experiment, never assumes that improvement was achieved."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse,json
from pathlib import Path
from aura.config import Config
from aura.self_improvement.search_strategy import ImprovementSearch
from aura.self_improvement.proposer import ModificationProposer
from aura.self_improvement.modification_registry import apply_modification
from aura.self_improvement.runner import evaluate_candidate
from aura.self_improvement.candidate_archive import CandidateArchive

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/cpu_smoke.yaml')
    p.add_argument('--steps',type=int,default=30);p.add_argument('--replicates',type=int,default=5)
    args=p.parse_args();c=Config.read(args.config)
    search=ImprovementSearch(c.seed);proposer=ModificationProposer(search)
    archive=CandidateArchive(Path(c.log_dir)/'candidate_lineage.jsonl')
    parent='v0'; reports=[]
    for round_no in range(2):
        name,mod=proposer.propose(parent,weakness=1)
        version=f'experiment-{len(archive.versions)}'
        report=evaluate_candidate(c,mod,steps=args.steps,replicates=args.replicates,version=version)
        archive.record(version,parent,mod,report)
        search.update(name,report.metrics['mean_prediction_gain'],report.total_cost['trained_model_steps'])
        reports.append({'round':round_no+1,'operator':name,'parent':parent,'version':version,
                        'accepted':report.accepted,'reason':report.reason,
                        'improvement_ci95':report.confidence_intervals['prediction_gain_ci95']})
        if not report.accepted: break  # no second-generation claims from a failed first gate
        parent=version;c=apply_modification(c,mod)
    result={'recursive_improvement_demonstrated':len(reports)>1 and all(r['accepted'] for r in reports),
            'rounds':reports,'lineage':str(archive.path)}
    print(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__':main()
