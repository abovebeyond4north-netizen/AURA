"""Reproduce the frozen Section 09 local acceptance matrix on real v44 gridworlds."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from aura.cognition.grid_door_study import replay
from aura.cognition.hypotheses import HypothesisEngine
from aura.cognition.intervention_study import run_causal_study
from aura.environments.causal_workbench import CausalWorkbench
from aura.environments.procedural_world import ProceduralWorld
from aura.memory.evidence_ledger import EvidenceLedger

PROTOCOL = Path(__file__).resolve().parents[1] / 'verification/section09/protocol.json'


class EvaluationCamera:
    """Expose only gym reset/step/close. Keep mechanism and diagnostics with the scorer."""
    def __init__(self, world, case, audit):
        self._world = world
        self._case = case
        self._audit = audit

    def reset(self, *, seed):
        self._seed = seed
        w = self._world
        image, _ = w.reset(seed=seed)
        if 'door_column' in self._case:
            w.walls.add(w.door)
            w.door = (w.size//2, self._case['door_column'])
            w.walls.remove(w.door)
            image = w._observe()
        return self._view(image), {}

    def _view(self, image):
        return np.zeros_like(image) if self._case.get('mask') else image

    def step(self, action):
        image, reward, done, truncated, _ = self._world.step(action)
        return self._view(image), reward, done, truncated, {}

    def close(self):
        state = self._world.diagnostic_state()
        self._audit[self._seed] = {'opened': state['door_open'], 'key': state['has_key'],
                                   'switch': state['switch_on'], 'rule': state['rule']}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def audit_ledger(ledger, reference, study_id, per_arm):
    """Reconcile confirmed cell counts independently and replay every observation."""
    require(ledger.verify(), 'Broken ledger chain')
    rows = ledger.records()
    by_id = {r['record_id']: r for r in rows}
    cells = np.zeros((4,3),dtype=int)
    seeds = set()
    counters = Counter()
    schedule = by_id[study_id+':design']['payload']
    for row in rows:
        if row['kind'] != 'door_measurement':
            continue
        m = row['payload']
        data = by_id[m['observation_id']]
        assignment = by_id[m['assignment_id']]['payload']
        require(row['source'] == 'INFERRED' and data['source'] == 'OBSERVED', 'Incorrect provenance')
        arm = assignment['arm']
        trace = data['payload']
        seed = trace['seed']
        require(seed not in seeds and seed == schedule['seed_schedule'][assignment['trial']], 'Reset seed discrepancy')
        seeds.add(seed)
        require(replay(trace, arm) == (m['opened'],m['missing_reason']), 'Trace does not replay')
        counters['traces_replayed'] += 1
        if m['opened'] is not None:
            oracle = reference[seed]
            require(m['opened'] == oracle['opened'], 'Physical passage/door-state discrepancy')
            require(oracle['key'] == bool(arm & 2) and oracle['switch'] == bool(arm & 1),
                    'Manipulation did not realize the assigned exposures')
            counters['independent_state_matches'] += 1
        if assignment['phase'] == 'confirm':
            j = assignment['trial'] - schedule['coupled'] - schedule['discovery']
            require(arm == schedule['confirm_schedule'][j], 'Unscheduled confirmation assignment')
            cells[arm, 2 if m['opened'] is None else int(m['opened'])] += 1
    result = by_id[study_id+':result']['payload']
    require(cells.tolist() == result['counts_closed_open_missing'], 'Confirmation counts disagree')
    require((cells.sum(axis=1)==per_arm).all(), 'Unbalanced confirmation')
    require(len(seeds)==len(reference)==len(schedule['seed_schedule']), 'Lost observed trials')
    for row in rows:
        if row['kind'] == 'hypothesis':
            h = row['payload']
            for rid in h['evidence_ids'] + h['counterevidence_ids']:
                require(by_id[rid]['source']=='OBSERVED' and by_id[rid]['kind']=='door_observation',
                        'Unresolvable hypothesis reference')
    return dict(counters)


def run_case(case, study_seed, proto, output):
    audit = {}
    def factory():
        w = ProceduralWorld(case['task'], size=case['size'], noise=case['noise'],
                            max_steps=80, partial=True, diagnostics=True)
        return EvaluationCamera(w, case, audit)

    study_id = f"{case['case']}-{study_seed}"
    # No case or task label is passed into the competing models.
    count = proto['mixture_confirmation_per_arm'] if case['case']=='mixed' else proto['confirmation_per_arm']
    ledger = EvidenceLedger(output / (study_id+'.sqlite'))
    try:
        result = HypothesisEngine().investigate_door(
            factory, ledger, study_id='anonymous', seed=study_seed,
            confirmation_per_arm=count, coupled_trials=proto['coupled_trials'],
            discovery_trials=proto['discovery_trials'], alpha=proto['per_study_familywise_alpha'],
            tolerance=proto['allowed_response_error'])
        checked = audit_ledger(ledger, audit, 'anonymous', count)
        require(result['status'] == case['expected'], f"{study_id} obtained {result['status']}")
        if case['case'].startswith('classic'):
            require(np.ptp(result['after_coupled_weights'][:4]) < 1e-12, 'Coupled observations improperly identified a rule')
            p = result['after_coupled_allocation']
            require(min(p[1],p[2]) > max(p[0],p[3]), 'Selector failed to prefer separating designs')
        return {'case':case['case'], 'seed':study_seed, 'expected':case['expected'],
                'evidence_file':study_id+'.sqlite', **checked, **result}
    finally:
        ledger.close()


def workbench(seed):
    passive = run_causal_study(CausalWorkbench('confounded',seed), 1200,seed,randomized=False)
    control = run_causal_study(CausalWorkbench('confounded',seed), 1200,seed)
    effect = run_causal_study(CausalWorkbench('effective',seed), 1200,seed)
    require(passive['result']=='NOT_IDENTIFIED' and passive['estimate']['association']>.4,
            'Passive confounding control failed')
    require(control['result']=='NEGLIGIBLE_EFFECT_SUPPORTED' and effect['result']=='POSITIVE_EFFECT_SUPPORTED',
            'Randomized confounding control failed')
    return {'seed':seed,'passive':passive,'randomized_null':control,'randomized_effect':effect}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replicate',action='store_true',help='Use the frozen fresh-seed replication')
    args = parser.parse_args()
    proto = json.loads(PROTOCOL.read_text())
    replication = PROTOCOL.with_name('replication.json')
    if args.replicate:
        proto['runs'] = json.loads(replication.read_text())['runs']
    args.output.mkdir(parents=True,exist_ok=False)
    rows = []
    for case in proto['families']:
        for seed in proto['runs']:
            row = run_case(case, seed, proto, args.output)
            rows.append(row)
            print(row['case'],seed,row['status'],row['counts_closed_open_missing'],flush=True)
    report = {'protocol_sha256':hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
              'replication_sha256':hashlib.sha256(replication.read_bytes()).hexdigest() if args.replicate else None,
              'scope':'local public controlled evidence; three paired design seeds per case',
              'cases':rows, 'workbench':[workbench(seed) for seed in proto['runs']], 'passed':True}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('PASS',len(rows),'gridworld studies and',len(report['workbench']),'confounding comparisons')


if __name__=='__main__':
    main()
