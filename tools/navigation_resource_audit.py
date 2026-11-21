"""Evidence-constrained resource accounting for AURA's public navigation studies.

Audits *observed* environment actions, successes/failures, and internal probes.
Historical rows do not contain total training time, CPU use, model updates,
or complete development/search compute. Unknown cost is not zero.

Never promote a section or claim independent evaluation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from tools.paired_cluster_inference import audit_public, audit_gate

MAX_ACTIONS = 65


def _valid_integer(value: object, description: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f'{description} must be a nonnegative integer')
    return value


def _pair_key(row: dict) -> tuple:
    return (row['seed'], row['size'], row['column'], row['reverse'], row['noise'])


def account(rows: list[dict], expected_arms: tuple[str, ...]) -> dict:
    """Account all actions including failures, without double-counting probes.

    The caller authenticates completeness to the immutable experiment manifest.
    This function also rejects missing, duplicate or unpaired controller records.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError('Missing raw resource evidence')
    arms = set(expected_arms)
    if len(arms) != len(expected_arms) or not arms:
        raise ValueError('Invalid controller specification')
    grouped = defaultdict(dict)
    per_arm = {
        arm: {
            'trials': 0, 'successes': 0, 'failures': 0,
            'environment_actions': 0, 'success_trial_actions': 0,
            'failure_trial_actions': 0, 'internal_probe_actions_included': 0,
            'recovery_attempts_reported': 0, 'resolution_events_reported': 0,
            'seed_groups': {},
        } for arm in expected_arms
    }
    for row in rows:
        if not isinstance(row, dict) or row.get('controller') not in arms:
            raise ValueError('Malformed/unexpected controller record')
        if type(row.get('success')) is not bool:
            raise ValueError('Missing explicit boolean outcome')
        actions = row.get('actions')
        if (not isinstance(actions, list) or len(actions) > MAX_ACTIONS or
                any(type(act) is not int or act not in range(5) for act in actions)):
            raise ValueError('Invalid recorded action trace')
        seed = row.get('seed')
        if type(seed) is not int or seed < 0:
            raise ValueError('Missing valid seed')
        if not isinstance(row.get('pose_errors'), list):
            raise ValueError('Missing localization audit')
        probes = _valid_integer(row.get('probe_actions'), 'probe_actions')
        recoveries = _valid_integer(row.get('recoveries'), 'recoveries')
        resolutions = _valid_integer(row.get('resolutions'), 'resolutions')
        if probes > len(actions):
            raise ValueError('Probe actions exceed total recorded actions')
        key = _pair_key(row)
        if row['controller'] in grouped[key]:
            raise ValueError('Duplicate controller trial')
        grouped[key][row['controller']] = (row['success'], len(actions))
        rec = per_arm[row['controller']]
        rec['trials'] += 1
        rec['successes' if row['success'] else 'failures'] += 1
        rec['environment_actions'] += len(actions)
        rec['success_trial_actions' if row['success'] else 'failure_trial_actions'] += len(actions)
        rec['internal_probe_actions_included'] += probes
        rec['recovery_attempts_reported'] += recoveries
        rec['resolution_events_reported'] += resolutions
        s = rec['seed_groups'].setdefault(
            str(seed), {'trials': 0, 'successes': 0, 'environment_actions': 0})
        s['trials'] += 1
        s['successes'] += int(row['success'])
        s['environment_actions'] += len(actions)
    for pair, samples in grouped.items():
        if set(samples) != arms:
            raise ValueError(f'Incomplete paired resource records: {pair}')
    for rec in per_arm.values():
        if rec['environment_actions'] != (rec['success_trial_actions'] +
                                          rec['failure_trial_actions']):
            raise ValueError('Action reconciliation failure')
        rec['environment_actions_per_success_including_failures'] = (
            rec['environment_actions'] / rec['successes'] if rec['successes'] else None)
        rec['goal_successes_per_1000_environment_actions'] = (
            1000 * rec['successes'] / rec['environment_actions']
            if rec['environment_actions'] else None)
        rec['action_budget_available'] = MAX_ACTIONS * rec['trials']
    arm_names = list(expected_arms)
    comparisons = {}
    for candidate in arm_names[1:]:
        baseline = arm_names[0]
        diff_by_seed = defaultdict(
            lambda: {'environment_actions': 0, 'goal_successes': 0, 'conditions': 0})
        for key, sampled in grouped.items():
            seed = str(key[0])
            b, c = sampled[baseline], sampled[candidate]
            diff_by_seed[seed]['environment_actions'] += c[1] - b[1]
            diff_by_seed[seed]['goal_successes'] += int(c[0]) - int(b[0])
            diff_by_seed[seed]['conditions'] += 1
        comparisons[f'{candidate}_minus_{baseline}'] = {
            'per_seed': dict(sorted(diff_by_seed.items())),
            'environment_actions': (per_arm[candidate]['environment_actions'] -
                                    per_arm[baseline]['environment_actions']),
            'goal_successes': (per_arm[candidate]['successes'] -
                               per_arm[baseline]['successes']),
        }
    return {
        'controllers': per_arm, 'paired_comparisons': comparisons,
        'paired_conditions': len(grouped), 'action_cap_per_trial': MAX_ACTIONS,
        'resources_not_recorded': {
            'model_training_updates': None, 'training_cpu_seconds': None,
            'training_wall_seconds': None, 'development_probe_cost': None,
            'full_controller_search_cost': None, 'training_memory_peak_bytes': None,
            'policy_selection_optimization_cost': None,
        },
        'internal_probe_counting': (
            'probe_actions are already inside environment_actions; never add them twice'),
        'interpretation': (
            'Identical per-trial action cap is not equal observed action or total training/compute cost.'),
        'evaluator_independent': False, 'acceptance_promoted': False,
    }


def audit_studies(manifest_bytes: bytes, public_bytes: bytes, gated_bytes: bytes) -> dict:
    public = json.loads(public_bytes)
    gated = json.loads(gated_bytes)
    audit_public(manifest_bytes, public)
    audit_gate(gated)
    return {
        'protocol': 'AURA-resource-accounting-v1',
        'wbs_ids': ['AURA-24.3-T2-S1', 'AURA-24.3-T2-S2', 'AURA-08.4-T1-S1'],
        'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'public_sha256': hashlib.sha256(public_bytes).hexdigest(),
        'gated_sha256': hashlib.sha256(gated_bytes).hexdigest(),
        'studies': {
            'public': account(public['rows'], (
                'EfficientRouteCamera', 'PersistentPoseCamera', 'ComposedRouteCamera')),
            'gated': account(gated['rows'], (
                'PersistentPoseCamera', 'ComposedRouteCamera', 'ConfidenceGatedRouteCamera')),
        },
        'raw_input_verified_against_saved_summary': True,
        'independent': False, 'acceptance_promoted': False,
        'unverified_budget_components': (
            'Historical traces do not disclose total controller development, '
            'training, measured CPU or wall-time.'),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--public', type=Path, required=True)
    p.add_argument('--gated', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = audit_studies(a.manifest.read_bytes(), a.public.read_bytes(),
                           a.gated.read_bytes())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write('\n')
    for name, study in report['studies'].items():
        print(name, [(ctrl, item['successes'], item['environment_actions'])
                     for ctrl, item in study['controllers'].items()])
    print('Total development cost NOT MEASURED; no acceptance promotion or independent validation')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
