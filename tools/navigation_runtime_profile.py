"""Public-only measured full-episode CPU and wall time in paired environments.

Includes environment creation, observations and controller execution; excludes
training, tuning, controller search, data generation and prior development.
Measurements vary by machine, load, threading, warmups and execution order.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

CONTROLLERS = (
    'PersistentPoseCamera', 'ComposedRouteCamera', 'ConfidenceGatedRouteCamera')


def validate_manifest(m: dict) -> None:
    if (m.get('protocol') != 'AURA-public-resource-profile-v1' or
            m.get('independent') is not False):
        raise ValueError('Only controlled public evaluation is accepted')
    if m.get('controllers') != list(CONTROLLERS) or m.get('action_cap') != 65:
        raise ValueError('Unexpected policy order or budget')
    seeds = m.get('seeds')
    if (not isinstance(seeds, list) or len(seeds) < 2 or
            any(type(x) is not int or x < 0 for x in seeds) or
            len(set(seeds)) != len(seeds)):
        raise ValueError('Invalid seed set')
    layouts = m.get('layouts')
    if (not isinstance(layouts, list) or not layouts or
            len(set(tuple(x) for x in layouts)) != len(layouts)):
        raise ValueError('Invalid layouts')
    if any(len(x) != 2 or any(type(v) is not int for v in x) or
           x[0] < 7 or x[0] % 2 != 1 or
           not 1 <= x[1] < x[0]-1 for x in layouts):
        raise ValueError('Invalid geometry')
    if m.get('reverse_after_key') != [False, True]:
        raise ValueError('Missing changed movement regime')
    noise = m.get('observation_noise')
    if (not isinstance(noise, list) or not noise or
            any(type(v) not in (int, float) or not 0 <= v <= 0.5 for v in noise) or
            len(set(noise)) != len(noise)):
        raise ValueError('Invalid observation corruption')


def validate(rows: list[dict], m: dict) -> dict:
    from tools.navigation_resource_audit import account
    validate_manifest(m)
    expected = {
        (seed, size, col, rev, noise, arm)
        for seed in m['seeds'] for size, col in m['layouts']
        for rev in m['reverse_after_key'] for noise in m['observation_noise']
        for arm in CONTROLLERS
    }
    seen = Counter((r['seed'], r['size'], r['column'], r['reverse'],
                    r['noise'], r['controller']) for r in rows)
    if set(seen) != expected or any(n != 1 for n in seen.values()):
        raise ValueError('Unexpected, omitted, or repeated runtime profiles')
    for row in rows:
        for metric in ('full_episode_wall_ns', 'full_episode_cpu_ns'):
            if type(row.get(metric)) is not int or row[metric] <= 0:
                raise ValueError(f'Invalid measured {metric}')
        if (type(row.get('execution_order')) is not int or
                not 0 <= row['execution_order'] < len(CONTROLLERS)):
            raise ValueError('Invalid execution order')
    report = account(rows, CONTROLLERS)
    for arm, sums in report['controllers'].items():
        selected = [r for r in rows if r['controller'] == arm]
        sums['measured_full_episode_wall_seconds'] = (
            sum(r['full_episode_wall_ns'] for r in selected) / 1e9)
        sums['measured_full_episode_cpu_seconds'] = (
            sum(r['full_episode_cpu_ns'] for r in selected) / 1e9)
        sums['wall_seconds_per_goal_including_failures'] = (
            sums['measured_full_episode_wall_seconds'] / sums['successes']
            if sums['successes'] else None)
        sums['cpu_seconds_per_goal_including_failures'] = (
            sums['measured_full_episode_cpu_seconds'] / sums['successes']
            if sums['successes'] else None)
    return report


def run(m: dict) -> list[dict]:
    validate_manifest(m)
    from experiments.persistent_pose_study import trial
    from aura.exploration.persistent_pose import PersistentPoseCamera
    from aura.exploration.composed_routes import ComposedRouteCamera
    from aura.exploration.confidence_gated_routes import ConfidenceGatedRouteCamera
    controllers = (
        PersistentPoseCamera, ComposedRouteCamera, ConfidenceGatedRouteCamera)
    rows = []
    for seed in m['seeds']:
        for size, col in m['layouts']:
            for rev in m['reverse_after_key']:
                for noise in m['observation_noise']:
                    # Rotate policy order; not proof of unbiased or blinded timing.
                    shift = (seed + size + col + int(rev) + int(noise * 1000)) % 3
                    for execution_order in range(3):
                        cls = controllers[(execution_order + shift) % 3]
                        wall = time.perf_counter_ns()
                        cpu = time.process_time_ns()
                        record = trial(seed, size, col, rev, noise, cls)
                        cpu_ns = time.process_time_ns() - cpu
                        wall_ns = time.perf_counter_ns() - wall
                        record.update(
                            full_episode_wall_ns=wall_ns,
                            full_episode_cpu_ns=cpu_ns,
                            execution_order=execution_order)
                        rows.append(record)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    m = json.loads(manifest_bytes)
    rows = run(m)
    report = validate(rows, m)
    result = {
        'protocol': m['protocol'],
        'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'source_commit': os.environ.get('GITHUB_SHA', 'local-unverified'),
        'metadata': {
            'python': sys.version.split()[0],
            'machine': platform.machine(),
            'platform': platform.platform(),
            'timing_scope': 'whole single-threaded trial call, not training or search',
            'measured_at_execution': True,
            'independent': False,
        },
        'rows': rows, 'summary': report,
        'acceptance_promoted': False, 'independent': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    for arm, counts in report['controllers'].items():
        print(arm, counts['successes'], counts['environment_actions'],
              round(counts['measured_full_episode_wall_seconds'], 3),
              round(counts['measured_full_episode_cpu_seconds'], 3))
    print('Development, model training, and search costs NOT MEASURED; no independent validation')


if __name__ == '__main__':
    main()
