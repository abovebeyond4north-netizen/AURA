"""Seed-clustered paired audit of public Section 08 navigation experiments.

One random seed is a cluster: layout/noise/movement trials reuse its generator.
Intervals and sign-flip enumerations are exploratory diagnostics, NOT a
confirmatory p-value, evaluator independence, or an acceptance promotion.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import random
from collections import defaultdict

PUBLIC_ARMS = ('EfficientRouteCamera', 'PersistentPoseCamera', 'ComposedRouteCamera')
GATED_ARMS = ('PersistentPoseCamera', 'ComposedRouteCamera', 'ConfidenceGatedRouteCamera')


def _trial_key(row: dict) -> tuple:
    fields = ('seed', 'size', 'column', 'reverse', 'noise')
    if any(field not in row for field in fields):
        raise ValueError('Missing paired trial keys')
    if type(row['seed']) is not int or type(row['size']) is not int or type(row['column']) is not int:
        raise ValueError('Malformed seed or geometry')
    if type(row['reverse']) is not bool or type(row['noise']) not in (float, int):
        raise ValueError('Malformed movement/noise')
    return tuple(row[field] for field in fields)


def _check_trials(rows: list, arms: tuple[str, ...]) -> dict[int, dict[tuple, dict[str, dict]]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError('No trial evidence')
    result: dict[int, dict[tuple, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if not isinstance(r, dict):
            raise ValueError('Malformed trial')
        trial = _trial_key(r)
        name = r.get('controller')
        if name not in arms:
            raise ValueError('Unknown or unexpected arm')
        if type(r.get('success')) is not bool:
            raise ValueError('Success must be an explicit boolean')
        actions = r.get('actions')
        if not isinstance(actions, list) or len(actions) > 65 or any(type(a) is not int or a not in range(5) for a in actions):
            raise ValueError('Invalid action trace or trial budget')
        if not isinstance(r.get('pose_errors'), list):
            raise ValueError('Invalid localization audit')
        arms_here = result[r['seed']][trial[1:]]
        if name in arms_here:
            raise ValueError('Duplicate controller trial')
        arms_here[name] = r
    expected_conditions = None
    for seed, conditions in result.items():
        if not conditions:
            raise ValueError('Empty seed cluster')
        for records in conditions.values():
            if set(records) != set(arms):
                raise ValueError('Missing controller from paired trial')
        if expected_conditions is None:
            expected_conditions = set(conditions)
        elif set(conditions) != expected_conditions:
            raise ValueError('Seeds do not have matched conditions')
    return result


def _quantile(values: list[float], q: float) -> float:
    arr = sorted(values)
    position = (len(arr) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    return arr[lo] + (position - lo) * (arr[hi] - arr[lo])


def _cluster_bootstrap(effects: list[float]) -> dict:
    n = len(effects)
    if n < 2:
        raise ValueError('At least two independently seeded clusters required')
    if n <= 7 and n ** n <= 100000:
        means = [sum(effects[i] for i in sample) / n for sample in product(range(n), repeat=n)]
        sampling = 'all possible seed-cluster resamples (with replacement)'
    else:
        rng = random.Random(264002)
        means = [sum(effects[rng.randrange(n)] for _ in range(n)) / n for _ in range(20000)]
        sampling = '20,000 fixed-seed cluster bootstrap draws'
    observed = sum(effects) / n
    # Sign flip is a *descriptive sensitivity check* only. Controller choices
    # were not randomized and experiments were examined repeatedly.
    if n <= 14:
        flips = [sum(effect * sign for effect, sign in zip(effects, signs)) / n
                 for signs in product((-1, 1), repeat=n)]
        sensitivity = sum(abs(x) >= abs(observed) - 1e-12 for x in flips) / len(flips)
    else:
        sensitivity = None
    return {'seed_clusters': n, 'observed_mean_success_difference': observed,
            'cluster_bootstrap_percentile_95': [_quantile(means, .025), _quantile(means, .975)],
            'bootstrap_sampling': sampling,
            'sign_flip_two_sided_sensitivity': sensitivity,
            'not_a_confirmatory_p_value': True,
            'independent_replications_insufficient': n < 8,
            'correlated_conditions_warning': 'Each seed repeats layout, movement and corruption conditions.'}


def paired_effect(rows: list, baseline: str, candidate: str, arms: tuple[str, ...]) -> dict:
    if baseline == candidate or baseline not in arms or candidate not in arms:
        raise ValueError('Invalid comparison')
    trials = _check_trials(rows, arms)
    per_seed = []
    improved = regressed = equal_success = equal_failure = 0
    for seed in sorted(trials):
        conditions = trials[seed]
        b_wins = sum(records[baseline]['success'] for records in conditions.values())
        c_wins = sum(records[candidate]['success'] for records in conditions.values())
        gains = losses = both = neither = 0
        for records in conditions.values():
            b = records[baseline]['success']
            c = records[candidate]['success']
            gains += int(c and not b)
            losses += int(b and not c)
            both += int(b and c)
            neither += int(not b and not c)
        improved += gains
        regressed += losses
        equal_success += both
        equal_failure += neither
        per_seed.append({'seed': seed, 'paired_conditions': len(conditions),
                         'baseline_successes': b_wins, 'candidate_successes': c_wins,
                         'improved': gains, 'regressed': losses,
                         'candidate_minus_baseline': (c_wins - b_wins)/len(conditions)})
    inference = _cluster_bootstrap([v['candidate_minus_baseline'] for v in per_seed])
    pairs = sum(v['paired_conditions'] for v in per_seed)
    if improved + regressed + equal_success + equal_failure != pairs:
        raise ValueError('Paired outcome accounting mismatch')
    return {'baseline': baseline, 'candidate': candidate, 'pairs': pairs,
            'paired': {'candidate_only': improved, 'baseline_only': regressed,
                       'both_successful': equal_success, 'both_failed': equal_failure,
                       'net_candidate_minus_baseline': improved - regressed},
            'per_seed': per_seed, 'exploratory_inference': inference}


def audit_public(manifest: bytes, result: dict) -> dict:
    from tools.navigation_audit import audit
    audit(manifest, result)  # checks manifest hash, completeness and source summary
    return {'protocol': 'AURA-public-confirmation-v1',
            'manifest_sha256': hashlib.sha256(manifest).hexdigest(),
            'comparisons': [paired_effect(result['rows'], 'PersistentPoseCamera', 'ComposedRouteCamera', PUBLIC_ARMS),
                            paired_effect(result['rows'], 'PersistentPoseCamera', 'EfficientRouteCamera', PUBLIC_ARMS)]}


def audit_gate(result: dict) -> dict:
    from experiments.confidence_gate_study import summarize
    if result.get('protocol') != 'AURA-public-confidence-gate-v1' or result.get('independent') is not False or result.get('acceptance_promoted') is not False:
        raise ValueError('Non-public or promoted gate evidence rejected')
    if summarize(result.get('rows', [])) != result.get('summary'):
        raise ValueError('Corrupt or inconsistent confidence-gate summary')
    if not all(type(v) is int for v in result.get('seeds', [])):
        raise ValueError('Malformed seeds')
    return {'protocol': result['protocol'],
            'source_commit': result.get('source_commit', 'unknown'),
            'comparisons': [paired_effect(result['rows'], 'PersistentPoseCamera', 'ComposedRouteCamera', GATED_ARMS),
                            paired_effect(result['rows'], 'ComposedRouteCamera', 'ConfidenceGatedRouteCamera', GATED_ARMS)]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--public', type=Path, required=True)
    ap.add_argument('--gated', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Refusing to replace previous evidence')
    pbytes = args.public.read_bytes()
    gbytes = args.gated.read_bytes()
    result = {
        'protocol': 'AURA-seed-cluster-exploratory-v1',
        'tasks': ['AURA-24.3-T1-S1', 'AURA-24.3-T1-S2', 'AURA-08.3-T2-S2', 'AURA-24.1-T2-S2'],
        'public_result_sha256': hashlib.sha256(pbytes).hexdigest(),
        'gate_result_sha256': hashlib.sha256(gbytes).hexdigest(),
        'studies': [audit_public(args.manifest.read_bytes(), json.loads(pbytes)),
                    audit_gate(json.loads(gbytes))],
        'evaluator_independent': False, 'acceptance_promoted': False,
        'interpretation': 'Four seed clusters, repeatedly tested public regimes, and adaptively developed policies. '
                          'Intervals and sign-flip scores are exploratory; no independent validation or '
                          'experiment-wise multiplicity correction. Do not pool studies as blinded replicates.'
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    for study in result['studies']:
        for comparison in study['comparisons']:
            x = comparison['exploratory_inference']
            print(f"{study['protocol']}: {comparison['candidate']} vs {comparison['baseline']}: "
                  f"difference={x['observed_mean_success_difference']:.4f}, "
                  f"cluster CI={x['cluster_bootstrap_percentile_95']}, "
                  f"exploratory signflip={x['sign_flip_two_sided_sensitivity']}, Nseed={x['seed_clusters']}")
    print('UNVERIFIED independence; NO acceptance promotion')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
