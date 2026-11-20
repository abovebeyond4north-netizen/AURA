"""Learn competing Boolean response models and confirm on a fresh factorial block."""
import math

import numpy as np
from scipy.special import betaincinv, xlogy

from aura.types import Hypothesis

ARMS = ('neither', 'switch_only', 'key_only', 'both')
# A constrained grammar over the two independently manipulable exposures.
KEY = np.array([False, False, True, True])
SWITCH = np.array([False, True, False, True])
MODELS = dict(KEY=KEY, SWITCH=SWITCH, AND=KEY & SWITCH, OR=KEY | SWITCH)
ASSUMPTIONS = [
    'Consistent reset population and no interference between randomized trials.',
    'The two preparation routes have no direct door effect other than the assigned exposures.',
    'Public RGB palette, cardinal motion, and independent physical passage check apply.',
    'Claims apply only to this context and these four interventions.'
]


class DoorCompetition:
    def __init__(self):
        self.patterns = np.stack(list(MODELS.values()))
        self.log_weights = np.zeros(len(MODELS) + 1)
        self.counts = np.zeros((4, 2), dtype=int)
        self.refs = {k: [[], []] for k in MODELS}

    @property
    def weights(self):
        weights = np.exp(self.log_weights - self.log_weights.max())
        return weights / weights.sum()

    def predictions(self):
        # The final model estimates an unconstrained response in each cell.
        other = (self.counts[:, 1] + 1) / (self.counts.sum(axis=1) + 2)
        return np.vstack((np.where(self.patterns, .95, .05), other))

    def design(self):
        probabilities = self.predictions()
        weights = self.weights
        def entropy(p):
            return -(xlogy(p, p) + xlogy(1-p, 1-p)) / math.log(2)
        information = np.maximum(0, entropy(weights @ probabilities) - weights @ entropy(probabilities))
        # Leave every experiment a nonzero probability as evidence accumulates.
        allocation = .9 * information / information.sum() + .025 if information.sum() > 1e-12 else np.full(4, .25)
        return information, allocation

    def observe(self, arm, outcome, evidence_id):
        if outcome is None:
            return
        p = self.predictions()[:, arm]
        self.log_weights += np.log(p if outcome else 1 - p)
        self.counts[arm, int(outcome)] += 1
        for name, pattern in MODELS.items():
            matched = bool(pattern[arm]) == outcome
            self.refs[name][0 if matched else 1].append(evidence_id)

    def hypotheses(self, context):
        info, allocation = self.design()
        for i, (name, pattern) in enumerate(MODELS.items()):
            yield Hypothesis(
                hypothesis_id=f'{context}:door:{name}',
                expression={'model': name, 'arms': list(ARMS), 'expected_open': pattern.tolist()},
                assumptions=list(ASSUMPTIONS),
                evidence_ids=list(self.refs[name][0]), counterevidence_ids=list(self.refs[name][1]),
                confidence=float(self.weights[i]), applicable_contexts=[context],
                proposed_tests=[{'arm': ARMS[j], 'information_bits': float(info[j]),
                                 'selection_probability': float(allocation[j])} for j in np.argsort(-info)]
            )


def confirm(counts, *, alpha=.05, tolerance=.25):
    """Exact simultaneous cell bounds. Count missing outcomes against both extremes."""
    if not 0 < alpha < 1 or not 0 < tolerance < .5:
        raise ValueError('Invalid confirmation parameters')
    counts = np.asarray(counts)
    if counts.shape != (4,3) or counts.dtype.kind not in 'iu' or (counts < 0).any():
        raise ValueError('Expected four nonnegative (closed, open, missing) count triples')
    intervals = []
    tail = alpha / (2 * len(ARMS))
    for closed, opened, missing in counts.tolist():
        n = closed + opened + missing
        high_success = opened + missing
        lo = float(betaincinv(opened, n-opened+1, tail)) if opened else 0.
        hi = float(betaincinv(high_success+1, n-high_success, 1-tail)) if high_success < n else 1.
        intervals.append([lo, hi])
    decisions = {}
    for name, pattern in MODELS.items():
        bands = [(1-tolerance, 1.) if bit else (0., tolerance) for bit in pattern]
        if all(l >= a and h <= b for (l,h),(a,b) in zip(intervals,bands)):
            decisions[name] = 'supported'
        elif any(h < a or l > b for (l,h),(a,b) in zip(intervals,bands)):
            decisions[name] = 'rejected'
        else:
            decisions[name] = 'unresolved'
    supported = [name for name, decision in decisions.items() if decision == 'supported']
    if len(supported) == 1:
        status = supported[0]
    elif all(x == 'rejected' for x in decisions.values()):
        status = 'ALL_CANDIDATES_REJECTED'
    else:
        status = 'INCONCLUSIVE'
    return {'status': status, 'decisions': decisions, 'arms': list(ARMS),
            'counts_closed_open_missing': counts.tolist(), 'simultaneous_intervals': intervals,
            'familywise_alpha': alpha, 'allowed_response_error': tolerance}
