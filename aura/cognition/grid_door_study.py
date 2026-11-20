"""Bounded active causal investigations in the primary environment.

The study is given an episode factory. Trial selection and interpretation
receive RGB images and the assigned actions only. Design, observations and
inferences have different ledger record types and sources.
"""
import base64
from collections import Counter
import hashlib
import zlib

import numpy as np

from aura.cognition.door_mechanisms import ARMS, MODELS, DoorCompetition, confirm
from aura.exploration.door_probe import Unmeasurable, measure_door
from aura.memory.semantic import SemanticMemory
from aura.types import Source

MAX_ACTIONS = 64


def _run_trial(factory, seed, arm):
    env = factory()
    actions, rewards, ended = [], [], []
    try:
        first, _ = env.reset(seed=seed)
        frames = [first.copy()]

        def step(action):
            if len(actions) >= MAX_ACTIONS:
                raise Unmeasurable('action budget exhausted')
            image, reward, terminated, truncated, _ = env.step(action)
            actions.append(action)
            rewards.append(float(reward))
            ended.append([bool(terminated), bool(truncated)])
            frames.append(image.copy())
            if terminated or truncated:
                raise Unmeasurable('episode ended before probe completed')
            return image

        outcome, reason = measure_door(first, step, arm)
    finally:
        if hasattr(env, 'close'):
            env.close()
    raw = np.stack(frames).tobytes()
    return outcome, reason, {
        'seed': seed, 'actions': actions, 'rewards': rewards, 'ended': ended,
        'frames_zlib_b64': base64.b64encode(zlib.compress(raw)).decode('ascii'),
        'frames_sha256': hashlib.sha256(raw).hexdigest()
    }


def replay(trial, arm):
    """Reconstruct the measurement from recorded pixels and verify chosen actions."""
    actions = trial['actions']
    raw = zlib.decompress(base64.b64decode(trial['frames_zlib_b64'], validate=True))
    if len(raw) != (len(actions)+1)*147 or hashlib.sha256(raw).hexdigest() != trial['frames_sha256']:
        raise ValueError('Invalid RGB trace')
    if len(trial['rewards']) != len(actions) or len(trial['ended']) != len(actions):
        raise ValueError('Incomplete recorded transitions')
    frames = np.frombuffer(raw, np.uint8).reshape(-1,3,7,7)
    pos = 0

    def step(action):
        nonlocal pos
        if pos == MAX_ACTIONS:
            raise Unmeasurable('action budget exhausted')
        if pos >= len(actions) or actions[pos] != action:
            raise ValueError('Recorded actions differ from the probe')
        pos += 1
        if any(trial['ended'][pos-1]):
            raise Unmeasurable('episode ended before probe completed')
        return frames[pos]

    outcome = measure_door(frames[0], step, arm)
    if pos != len(actions):
        raise ValueError('Trailing recorded actions')
    return outcome


def run(engine, factory, ledger, *, study_id, seed, discovery_trials=16,
        confirmation_per_arm=64, coupled_trials=8, alpha=.05, tolerance=.25):
    for n in (coupled_trials, discovery_trials, confirmation_per_arm):
        if type(n) is not int or not 0 <= n <= 1024:
            raise ValueError('Study counts must be integers in [0, 1024]')
    if not study_id or type(seed) is not int or seed < 0:
        raise ValueError('Expected a nonempty study ID and nonnegative integer seed')
    if not 0 < alpha < 1 or not 0 < tolerance < .5:
        raise ValueError('Invalid confirmation parameters')
    design_rng, confirm_rng, env_rng = (np.random.default_rng(s)
        for s in np.random.SeedSequence(seed).spawn(3))
    fixed = np.repeat(np.arange(4), confirmation_per_arm)
    confirm_rng.shuffle(fixed)
    total = coupled_trials + discovery_trials + len(fixed)
    seeds = env_rng.integers(0, 2**63-1, size=total).tolist()
    if len(set(seeds)) != len(seeds):
        raise ValueError('Reset seed collision')
    ledger.append(study_id+':design', 'door_design', Source.EXTERNALLY_SPECIFIED, {
        'study': study_id, 'seed': seed, 'coupled': coupled_trials, 'discovery': discovery_trials,
        'confirmation_per_arm': confirmation_per_arm, 'alpha': alpha, 'tolerance': tolerance,
        'model_truth_tables': {k:v.tolist() for k,v in MODELS.items()}, 'arms':list(ARMS),
        'max_actions_per_trial': MAX_ACTIONS, 'seed_schedule':seeds, 'confirm_schedule':fixed.tolist()
    })
    models = DoorCompetition()
    semantic = SemanticMemory(ledger)
    counts = np.zeros((4,3), dtype=int)
    costs = Counter()
    failures = Counter()
    discovery_arms = Counter()

    def revise():
        for hypothesis in models.hypotheses(study_id):
            semantic.revise(hypothesis)
            engine.hypotheses[hypothesis.hypothesis_id] = hypothesis

    def trial(phase, i, arm, assignment, bits=None):
        trace_id = f'{study_id}:trial:{i}'
        decision_id = trace_id+':assignment'
        ledger.append(decision_id, 'door_assignment', Source.INFERRED, {
            'study': study_id, 'phase': phase, 'trial': i, 'arm': arm,
            'design': assignment, 'information_bits': bits
        })
        outcome, reason, trace = _run_trial(factory, seeds[i], arm)
        ledger.append(trace_id, 'door_observation', Source.OBSERVED, trace)
        ledger.append(trace_id+':measurement', 'door_measurement', Source.INFERRED, {
            'observation_id': trace_id, 'assignment_id': decision_id, 'opened': outcome, 'missing_reason': reason
        })
        costs['trials'] += 1
        costs['actions'] += len(trace['actions'])
        if reason:
            failures[reason] += 1
        models.observe(arm, outcome, trace_id)
        if phase == 'confirm':
            counts[arm, 2 if outcome is None else int(outcome)] += 1

    revise()
    # The two exposures always coincide in this control. It cannot tell the
    # four rules apart, even with perfect readings and a strong association.
    for i in range(coupled_trials):
        trial('coupled', i, 3 * (i % 2), 'coupled protocol')
    coupled_weights = models.weights.tolist()
    bits, probabilities = models.design()
    first_allocation = probabilities.tolist()
    revise()
    for i in range(discovery_trials):
        bits, probabilities = models.design()
        arm = int(design_rng.choice(4, p=probabilities))
        discovery_arms[ARMS[arm]] += 1
        trial('discover', i+coupled_trials, arm, probabilities.tolist(), bits.tolist())
    frozen_weights = models.weights.tolist()
    revise()
    # Confirmation schedule was stored before seeing any observations.
    for j, arm in enumerate(fixed):
        trial('confirm', coupled_trials+discovery_trials+j, int(arm), 'frozen balanced random order')
    revise()
    report = {
        'study': study_id, **confirm(counts, alpha=alpha, tolerance=tolerance),
        'costs': dict(costs), 'missing_reasons_all_phases': dict(failures),
        'weight_order': [*MODELS,'OTHER'], 'after_coupled_weights': coupled_weights,
        'after_coupled_allocation': first_allocation, 'discovery_arm_counts': dict(discovery_arms),
        'preconfirmation_weights': frozen_weights
    }
    ledger.append(study_id+':result', 'door_result', Source.INFERRED, report)
    return report
