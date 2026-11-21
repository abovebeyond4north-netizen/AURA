"""Bounded image-goal planning with learned ensemble predictions.

Only already-observed RGB frames may be target states. Plans are hypothetical;
never write simulated images or latents to observed replay or evidence ledger.
This module does not execute a step or claim planning improves generalization.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class PlanProposal:
    status: str
    actions: tuple[int, ...]
    first_action: int | None
    score: float | None
    uncertainty: float | None
    evaluated_sequences: int
    reason: str
    source: str = 'IMAGINED_NON_EVIDENCE'


class BoundedModelPlanner:
    """Enumerate short hypothetical action sequences against an observed goal.

    Uses a WorldModel's trained encoder and recurrent dynamics ensemble. The
    scoring target MUST be a genuine previously observed RGB frame, supplied by
    the caller. This interface cannot independently authenticate target custody.
    """

    def __init__(self, model: Any, *, max_horizon: int = 2,
                 max_evaluations: int = 30, min_updates: int = 1,
                 uncertainty_weight: float = 0.1,
                 disagreement_ceiling: float | None = None):
        if type(max_horizon) is not int or not 1 <= max_horizon <= 3:
            raise ValueError('Horizon must be within 1–3 actions')
        if type(max_evaluations) is not int or not 1 <= max_evaluations <= 125:
            raise ValueError('Invalid compute quota')
        if type(min_updates) is not int or min_updates < 1:
            raise ValueError('Only models with real observed training updates may propose plans')
        if not isfinite(uncertainty_weight) or uncertainty_weight < 0:
            raise ValueError('Invalid uncertainty penalty')
        if disagreement_ceiling is not None and (
                not isfinite(disagreement_ceiling) or disagreement_ceiling < 0):
            raise ValueError('Invalid uncertainty limit')
        self.model = model
        self.max_horizon = max_horizon
        self.max_evaluations = max_evaluations
        self.min_updates = min_updates
        self.uncertainty_weight = float(uncertainty_weight)
        self.disagreement_ceiling = disagreement_ceiling

    @staticmethod
    def _frame(frame):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.shape != (3, 7, 7):
            raise ValueError('Only observed uint8 RGB 3x7x7 images may be scored')
        return frame

    def propose(self, current_observation: np.ndarray, goal_observation: np.ndarray,
                allowed_actions: tuple[int, ...] = (0,1,2,3), *,
                goal_source: str = 'OBSERVED') -> PlanProposal:
        if goal_source != 'OBSERVED':
            raise ValueError('An imagined or inferred target cannot masquerade as observed')
        current = self._frame(current_observation)
        goal = self._frame(goal_observation)
        if (not isinstance(allowed_actions,tuple) or not allowed_actions or
                len(set(allowed_actions)) != len(allowed_actions) or
                any(type(a) is not int or a not in range(5) for a in allowed_actions)):
            raise ValueError('Action mask must be a nonempty unique allowed-action tuple')
        if len(allowed_actions)**self.max_horizon > self.max_evaluations:
            return PlanProposal('abstain',(),None,None,None,0,'sequence evaluation quota exceeded')
        if getattr(self.model,'train_steps',0) < self.min_updates:
            return PlanProposal('abstain',(),None,None,None,0,'insufficient observed training updates')
        heads = getattr(self.model,'heads',None)
        if not heads:
            raise ValueError('An ensemble of model dynamics heads is required')
        old_training = bool(self.model.training)
        evaluated = 0
        best = None
        try:
            self.model.eval()
            with torch.no_grad():
                device = getattr(self.model,'device_name','cpu')
                x = torch.as_tensor(current,device=device,dtype=torch.float32).unsqueeze(0) / 255.
                target = torch.as_tensor(goal,device=device,dtype=torch.float32).unsqueeze(0) / 255.
                for actions in product(sorted(allowed_actions),repeat=self.max_horizon):
                    z = self.model.encoder(x)
                    hidden = [None] * len(heads)
                    risk = 0.
                    image = None
                    for action in actions:
                        action_tensor = torch.tensor([action],dtype=torch.long,device=device)
                        predictions = [head(z,action_tensor,hidden[i]) for i,head in enumerate(heads)]
                        latent = torch.stack([item[0] for item in predictions])
                        images = torch.stack([item[1] for item in predictions])
                        z = latent.mean(0)
                        hidden = [item[2] for item in predictions]
                        image = images.mean(0)
                        risk += float(images.var(0,unbiased=False).mean().item())
                    mse = float((image-target).square().mean().item())
                    mean_risk = risk/self.max_horizon
                    if not isfinite(mse) or not isfinite(mean_risk):
                        raise FloatingPointError('Non-finite imagined rollout')
                    evaluated += 1
                    if self.disagreement_ceiling is not None and mean_risk > self.disagreement_ceiling:
                        continue
                    score = mse + self.uncertainty_weight * mean_risk
                    key = (score,actions)
                    if best is None or key < best[0]:
                        best = key, mean_risk
        finally:
            self.model.train(old_training)
        if best is None:
            return PlanProposal('abstain',(),None,None,None,evaluated,'all rollouts exceed uncertainty limit')
        (_,actions),risk=best
        return PlanProposal('proposal',tuple(actions),actions[0],float(best[0][0]),risk,
                            evaluated,'hypothetical prediction only')


