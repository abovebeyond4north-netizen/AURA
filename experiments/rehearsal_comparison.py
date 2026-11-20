"""Compare matched public developmental runs, including their acquisition and sample costs."""
import argparse
import json
from pathlib import Path
from aura.memory.evidence_ledger import EvidenceLedger


def _read(directory):
    directory = Path(directory)
    study = json.loads((directory/'study.json').read_text())
    diagnostic = json.loads((directory/'lineage_diagnostic.json').read_text())
    ledger = EvidenceLedger(directory/'evidence.sqlite')
    try:
        if not ledger.verify():
            raise ValueError('Invalid evidence chain')
        reports = study['result']['imitation_candidates']
        if [r['payload'] for r in ledger.records('imitation_candidate')] != reports:
            raise ValueError('Candidate report mismatch')
        records = ledger.records('experience')
        index = {r['record_id']: r['payload'] for r in records}
        splits = [[[[index[eid] for eid in ep] for ep in r[key]]
                   for key in ('fresh_training_evidence','validation_evidence')] for r in reports]
        if len(diagnostic['generations']) != len(reports):
            raise ValueError('Diagnostic generation count mismatch')
        return study, diagnostic, [r['payload'] for r in records], splits
    finally:
        ledger.close()


def compare(fresh, rehearsal):
    a, da, ea, sa = _read(fresh)
    b, db, eb, sb = _read(rehearsal)
    ca, cb = a['config'].copy(), b['config'].copy()
    ca.pop('imitation_budgeted_rehearsal', False)
    bounded = cb.pop('imitation_budgeted_rehearsal', False)
    if ca.pop('imitation_rehearsal_episodes') != 0 or cb.pop('imitation_rehearsal_episodes') <= 0 or ca != cb:
        raise ValueError('Unmatched configuration')
    if ea != eb or sa != sb:
        raise ValueError('Unmatched observed acquisition or fresh splits')
    results = []
    for ga, gb in zip(da['generations'], db['generations'], strict=True):
        if ga['generation'] != gb['generation']:
            raise ValueError('Unmatched generation')
        pairs = list(zip(ga['rows'], gb['rows'], strict=True))
        for x,y in pairs:
            if any(x[k] != y[k] for k in ('case','size','noise','seed')):
                raise ValueError('Unmatched evaluation scenario')
        results.append({'generation': ga['generation'], 'episodes': len(pairs),
                        'fresh_successes': sum(x['success'] for x,y in pairs),
                        'rehearsal_successes': sum(y['success'] for x,y in pairs),
                        'rehearsal_only': sum(not x['success'] and y['success'] for x,y in pairs),
                        'fresh_only': sum(x['success'] and not y['success'] for x,y in pairs)})
    return {'scope':'public development diagnostic; matched experience and update counts; optimizer sample processing capped by fresh-only capacity' if bounded else 'public development diagnostic; matched experience and update counts, not matched sample processing; no promotion',
            'budgeted_rehearsal': bounded,
            'acquisition_payloads_equal':True,'fresh_split_payloads_equal':True,
            'generations':results,
            'costs':{name:{'training_transitions_presented':sum(r['training_transition_presentations'] for r in study['result']['imitation_candidates']),
                           'training_padded_steps_presented':sum(r['training_padded_step_presentations'] for r in study['result']['imitation_candidates']) if all('training_padded_step_presentations' in r for r in study['result']['imitation_candidates']) else None,
                           'optimizer_updates':study['result']['resource_metrics']['imitation_optimizer_updates'],
                           'acquisition_interactions':study['result']['resource_metrics']['environment_interactions'],
                           'total_online_interactions':study['result']['resource_metrics']['total_environment_interactions'],
                           'diagnostic_interactions':diag['diagnostic_interactions'],
                           'reload_interactions':diag['reload_interactions']}
                     for name,study,diag in [('fresh',a,da),('rehearsal',b,db)]}}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fresh', required=True, type=Path)
    p.add_argument('--rehearsal', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args=p.parse_args()
    report=compare(args.fresh,args.rehearsal)
    with args.output.open('x') as f: json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report,indent=2,allow_nan=False))
