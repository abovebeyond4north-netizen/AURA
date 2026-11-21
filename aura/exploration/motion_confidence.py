"""Observation-only action-effect confidence and transactional localization.

No privileged position, action mapping, noise probability, or evaluator state is
consumed. Beliefs are empirical categorical counts, not calibrated probabilities.
Unexpected evidence invalidates the current regime rather than silently
changing a trusted control map. Physical actions cannot be rolled back.
"""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import dataclass

import numpy as np

from aura.environments.procedural_world import MOVES
from aura.exploration.door_probe import Camera, Unmeasurable
from aura.exploration.change_dynamics import EvidenceChangeCamera

# These colored tiles are assumed physically fixed across ordinary movement.
# A collected key and an opened/disappearing door must NOT be used as anchors.
FIXED_ANCHORS = frozenset(('crate', 'switch', 'goal'))
DISPLACEMENTS = frozenset(MOVES)


def independent_visual_shift(before: np.ndarray, after: np.ndarray):
    """Abstain unless >=2 distinct persistent landmarks agree on one cardinal move.

    Unknown pixels, one visible anchor, disappearing anchors, and disagreement
    never become mapping observations. Does not read the simulator's position.
    """
    if (not isinstance(before, np.ndarray) or not isinstance(after, np.ndarray) or
        before.shape != (3, 7, 7) or after.shape != (3, 7, 7) or
        before.dtype != np.uint8 or after.dtype != np.uint8):
        raise ValueError('Requires paired uint8 RGB camera frames [3,7,7]')
    old, new = Camera.anchors(before), Camera.anchors(after)
    shared = old.keys() & new.keys() & FIXED_ANCHORS
    if len(shared) < 2:
        return None
    displacements = {(old[k][0] - new[k][0], old[k][1] - new[k][1]) for k in shared}
    if len(displacements) != 1:
        return None
    delta = next(iter(displacements))
    return delta if delta in DISPLACEMENTS else None


@dataclass(frozen=True)
class MotionBeliefSummary:
    regime: int
    observations: int
    committed_mapping: tuple[int, ...] | None
    action_support: tuple[int, ...]
    action_directions: tuple[int | None, ...]
    action_confidence: tuple[float, ...]
    abstentions: int
    contradictions: int
    rollbacks: int


class OnlineMotionConfidence:
    """Learn a raw-action->physical-direction bijection, fail closed on shifts.

    All counts come exclusively from camera anchor shifts. A candidate mapping
    is committed only when all four raw controls have repeated consistent
    evidence forming one bijection. Confidence uses a symmetric Dirichlet(1)
    posterior mean solely as a descriptive heuristic; it is not calibrated.
    """

    def __init__(self, min_evidence: int = 6, threshold: float = .65):
        if type(min_evidence) is not int or min_evidence < 2:
            raise ValueError('At least two separate movement observations required')
        if not isinstance(threshold, (int, float)) or not .5 < threshold < 1:
            raise ValueError('Invalid evidence-confidence threshold')
        self.min_evidence = min_evidence
        self.threshold = float(threshold)
        self.counts = {raw: Counter() for raw in range(4)}
        self.committed_mapping = None
        self.regime = 0
        self.observations = 0
        self.abstentions = 0
        self.contradictions = 0
        self.rollbacks = 0
        self.audit_events = []

    @staticmethod
    def _best(counts):
        if not counts:
            return None, 0.
        direction, votes = min(counts.items(), key=lambda x: (-x[1], x[0]))
        return direction, (votes + 1) / (sum(counts.values()) + 4)

    def summary(self):
        best = [self._best(self.counts[i]) for i in range(4)]
        return MotionBeliefSummary(
            regime=self.regime, observations=self.observations,
            committed_mapping=self.committed_mapping,
            action_support=tuple(sum(self.counts[i].values()) for i in range(4)),
            action_directions=tuple(x[0] for x in best),
            action_confidence=tuple(x[1] for x in best),
            abstentions=self.abstentions, contradictions=self.contradictions,
            rollbacks=self.rollbacks)

    def invalidate(self, reason: str):
        """Rollback confidence state; preserve a durable diagnostic event.

        Does not undo physical motion or mutate the camera's authoritative map.
        """
        if not isinstance(reason, str) or not reason:
            raise ValueError('Rollback must have a recorded reason')
        if any(self.counts.values()) or self.committed_mapping is not None:
            self.audit_events.append({'kind': 'rollback', 'regime': self.regime,
                                      'reason': reason, 'support': self.summary().action_support,
                                      'committed_mapping': self.committed_mapping})
            self.rollbacks += 1
            self.regime += 1
        self.counts = {raw: Counter() for raw in range(4)}
        self.committed_mapping = None

    def observe_frames(self, raw: int, before: np.ndarray, after: np.ndarray):
        if type(raw) is not int or not 0 <= raw < 4:
            raise ValueError('Raw control must be a cardinal action index')
        shift = independent_visual_shift(before, after)
        if shift is None:
            self.abstentions += 1
            return False
        direction = MOVES.index(shift)
        if self.committed_mapping is not None and self.committed_mapping[raw] != direction:
            self.contradictions += 1
            self.invalidate('observed movement contradicts committed mapping')
            # A single contradictory movement is not enough to trust a new rule.
            return False
        self.counts[raw][direction] += 1
        self.observations += 1
        # Every raw action must independently be supported, not only aggregate
        # observations from one direction. Opposite motion cannot be inferred
        # from an engineered assumption in this observational confidence layer.
        options = []
        for index in range(4):
            counts = self.counts[index]
            best, confidence = self._best(counts)
            if (best is None or sum(counts.values()) < self.min_evidence or
                    confidence < self.threshold):
                return True
            options.append(best)
        if len(set(options)) != 4:
            return True
        self.committed_mapping = tuple(options)
        return True


class ConfidenceRollbackCamera(EvidenceChangeCamera):
    """Existing change recovery plus separate observation-only confidence.

    Confidence is advisory; no unverified mapping is applied to navigation.
    Candidate mapping changes remain governed by the existing camera's
    conservative reidentification. Failed map commit is rolled back atomically.
    """

    def __init__(self, image, step):
        self.motion_confidence = OnlineMotionConfidence()
        self.failed_commits = 0
        super().__init__(image, step)

    def _commit_reidentified(self, mapping, pos, image):
        # Protect all shared localization state from exceptions in _observe.
        # Do NOT roll back physical world movement, only local hypotheses.
        snapshot = (self.mapping, self.position, self.image.copy(),
                    copy.deepcopy(self.map), copy.deepcopy(self.landmarks),
                    self.change_resolutions)
        try:
            super()._commit_reidentified(mapping, pos, image)
        except Exception:
            (self.mapping, self.position, self.image, self.map,
             self.landmarks, self.change_resolutions) = snapshot
            self.failed_commits += 1
            self.motion_confidence.invalidate('failed reidentification commit')
            raise

    def act(self, action, expected):
        if not self.monitoring or action >= 4:
            return super().act(action, expected)
        raw = self.mapping.index(action)
        prior = self.image.copy()
        resolutions = self.change_resolutions
        probes = self.change_probes
        try:
            result = super().act(action, expected)
        except Unmeasurable:
            self.motion_confidence.invalidate('unresolved observed movement')
            raise
        if self.change_resolutions != resolutions or self.change_probes != probes:
            # The camera may have performed several hidden-internal probes.
            # A before/after image then cannot be attributed to one raw action.
            self.motion_confidence.invalidate('multi-action reidentification')
        else:
            self.motion_confidence.observe_frames(raw, prior, self.image)
        return result
