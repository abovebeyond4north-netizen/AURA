"""Shared immutable contracts separating observed, inferred, imagined and specified data."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import numpy as np

class Source(str, Enum):
    OBSERVED = 'OBSERVED'
    INFERRED = 'INFERRED'
    IMAGINED = 'IMAGINED'
    EXTERNALLY_SPECIFIED = 'EXTERNALLY_SPECIFIED'

@dataclass(frozen=True)
class Experience:
    observation: np.ndarray
    action: int
    next_observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    episode_id: str
    evidence_id: str
    source: Source = Source.OBSERVED
    seed: int = 0
    context: str = 'key'

    def __post_init__(self):
        if self.source is not Source.OBSERVED:
            raise ValueError('Experience is reserved for observed environment transitions')
        if self.observation.shape != self.next_observation.shape:
            raise ValueError('Transition observations must have matching shapes')
        if not 0 <= self.action <= 4:
            raise ValueError('Action out of range')

@dataclass
class BeliefState:
    latent_state: Any
    uncertainty: float
    history_reference: str

@dataclass
class Hypothesis:
    hypothesis_id: str
    expression: dict[str, Any]
    assumptions: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    counterevidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5
    applicable_contexts: list[str] = field(default_factory=list)
    proposed_tests: list[dict] = field(default_factory=list)

@dataclass
class Skill:
    skill_id: str
    preconditions: dict[str, Any]
    policy_reference: str
    termination_condition: dict[str, Any]
    expected_effects: dict[str, Any]
    evidence_ids: list[str] = field(default_factory=list)
    successes: int = 0
    failures: int = 0
    applicable_contexts: list[str] = field(default_factory=list)

    @property
    def empirical_success_rate(self) -> float:
        return self.successes / max(1, self.successes + self.failures)

@dataclass(frozen=True)
class Modification:
    target: str
    operation: str
    parameters: dict[str, float | int]
    parent_version: str

@dataclass
class EvaluationReport:
    candidate_version: str
    baseline_version: str
    metrics: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]
    regressions: dict[str, float]
    total_cost: dict[str, float]
    accepted: bool
    reason: str
