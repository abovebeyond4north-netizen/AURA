"""Experimental multi-hypothesis camera localization; no hidden environment state.

This module does not replace the validated v50 controller. It deliberately
abstains when evidence cannot distinguish poses, rather than trusting commands.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from aura.exploration.visual_frontier import SHIFTS


@dataclass(frozen=True)
class PoseEstimate:
    shift: tuple[int, int] | None
    confidence: float
    probabilities: tuple[float, ...]


def _log_likelihood(before: np.ndarray, after: np.ndarray, shift, corruption: float):
    """Pixel-channel likelihood with explicit independent corruption model.

    Center-marker pixels are excluded. This is an engineered camera model,
    not a learned likelihood or a guarantee for correlated corruption.
    """
    if before.shape != (3, 7, 7) or after.shape != (3, 7, 7):
        raise ValueError('Expected channel-first 7x7 RGB')
    dy, dx = shift
    rr, cc = np.indices((7, 7))
    r, c = rr + dy, cc + dx
    valid = ((r >= 0) & (r < 7) & (c >= 0) & (c < 7)
             & ~((rr == 3) & (cc == 3)) & ~((r == 3) & (c == 3)))
    a = before[:, r[valid], c[valid]]
    b = after[:, rr[valid], cc[valid]]
    match = np.all(a == b, axis=0)
    # A corrupted RGB pixel can coincidentally match; conservative approximation.
    p_match = (1 - corruption) ** 2 + (1 - (1 - corruption) ** 2) / (256 ** 3)
    p_match = min(1 - 1e-9, max(1e-9, p_match))
    return float(match.sum() * math.log(p_match) + (~match).sum() * math.log1p(-p_match))


class PoseBelief:
    def __init__(self, corruption: float = .03, motion_prior: float = .8,
                 threshold: float = .995):
        if not 0 < corruption < .5 or not 0 < motion_prior < 1 or not .5 < threshold < 1:
            raise ValueError('Invalid probability parameters')
        self.corruption = corruption
        self.motion_prior = motion_prior
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.log_weights = np.full(len(SHIFTS), -math.log(len(SHIFTS)))

    def update(self, before, after, expected=None) -> PoseEstimate:
        before = np.asarray(before, dtype=np.uint8)
        after = np.asarray(after, dtype=np.uint8)
        if before.shape != (3, 7, 7) or after.shape != (3, 7, 7):
            raise ValueError('Expected channel-first 7x7 RGB')
        if expected is not None and expected not in SHIFTS:
            raise ValueError('Expected shift outside movement contract')
        # Per-transition posterior: do not carry shift weights across distinct
        # transitions (which would conflate current movement with past movement).
        prior = np.full(len(SHIFTS), (1-self.motion_prior)/(len(SHIFTS)-1))
        if expected is None:
            prior[:] = 1/len(SHIFTS)
        else:
            prior[SHIFTS.index(expected)] = self.motion_prior
        logits = np.log(prior) + np.array([
            _log_likelihood(before, after, shift, self.corruption) for shift in SHIFTS])
        weights = np.exp(logits - np.max(logits))
        weights /= weights.sum()
        k = int(np.argmax(weights))
        confidence = float(weights[k])
        return PoseEstimate(SHIFTS[k] if confidence >= self.threshold else None,
                            confidence, tuple(float(x) for x in weights))
