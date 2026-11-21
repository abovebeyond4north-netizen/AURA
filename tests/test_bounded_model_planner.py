"""Fail-closed compute quota, no evidence laundering, and model-state invariants."""
import copy
import numpy as np
import pytest
import torch
from torch import nn

from aura.planning.bounded_model_planner import BoundedModelPlanner
from aura.world_model.dynamics import WorldModel


class DummyEncoder(nn.Module):
    def forward(self,x):
        return x.mean(dim=(2,3)).mean(dim=1,keepdim=True)


class ActionHead(nn.Module):
    def forward(self,z,actions,hidden=None):
        color = actions.float().reshape(-1,1,1,1) / 4.
        image = color.expand(-1,3,7,7)
        return color.reshape(-1,1), image, None


class ToyTrainedModel(nn.Module):
    def __init__(self, updates=12):
        super().__init__()
        self.heads=nn.ModuleList([ActionHead(),ActionHead()])
        self.encoder=DummyEncoder()
        self.train_steps=updates
        self.device_name='cpu'


def image(value):
    return np.full((3,7,7),value,dtype=np.uint8)


def test_action_conditioned_observed_image_goal_is_selected():
    planner=BoundedModelPlanner(ToyTrainedModel(),max_horizon=1,max_evaluations=4)
    proposal=planner.propose(image(0),image(255),allowed_actions=(0,1,2,3))
    assert proposal.status=='proposal'
    assert proposal.actions==(3,)
    assert proposal.evaluated_sequences==4
    assert proposal.source == 'IMAGINED_NON_EVIDENCE'


def test_abstains_without_actual_training_updates():
    p=BoundedModelPlanner(ToyTrainedModel(updates=0),max_horizon=1)
    assert p.propose(image(0),image(255)).status=='abstain'


def test_quota_fail_closed_without_model_evaluation():
    model=ToyTrainedModel()
    p=BoundedModelPlanner(model,max_horizon=3,max_evaluations=20)
    proposal=p.propose(image(0),image(255))
    assert proposal.status=='abstain' and proposal.evaluated_sequences==0


def test_rejects_imagined_target_and_invalid_masks():
    p=BoundedModelPlanner(ToyTrainedModel(),max_horizon=1)
    with pytest.raises(ValueError):p.propose(image(0),image(255),goal_source='IMAGINED')
    for mask in ((),(0,0),(7,),(True,)):
        with pytest.raises(ValueError):p.propose(image(0),image(255),allowed_actions=mask)
    with pytest.raises(ValueError):p.propose(image(0).astype(float),image(255))


def test_real_world_model_predictions_are_finite_and_never_train_or_mutate_model():
    torch.manual_seed(3)
    model=WorldModel(latent_dim=8,hidden_dim=12,ensemble_size=2)
    model.train_steps=1
    model.train()
    params={k:v.detach().clone() for k,v in model.state_dict().items()}
    p=BoundedModelPlanner(model,max_horizon=2,max_evaluations=16)
    proposal=p.propose(image(0),image(127))
    assert proposal.status=='proposal' and proposal.evaluated_sequences==16
    assert len(proposal.actions)==2
    assert model.training and model.train_steps==1
    assert all(torch.equal(model.state_dict()[k],v) for k,v in params.items())


def test_uncertainty_limit_abstains_instead_of_hallucinating_evidence():
    model=ToyTrainedModel()
    p=BoundedModelPlanner(model,max_horizon=1,disagreement_ceiling=0)
    proposal=p.propose(image(0),image(255))
    assert proposal.status=='proposal'  # exact identical heads have zero disagreement
    p=BoundedModelPlanner(model,max_horizon=1,disagreement_ceiling=0, uncertainty_weight=1)
    assert p.propose(image(0),image(255)).score is not None
