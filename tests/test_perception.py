import numpy as np
import torch
from aura.perception.encoder import ObservationEncoder
from aura.world_model.dynamics import WorldModel
from protected_verifier.evaluator import collect

def test_encoder_shape_and_noncollapsed_initialization():
    x=torch.randint(0,256,(8,3,7,7),dtype=torch.uint8)
    e=ObservationEncoder(12)
    z=e(x)
    assert z.shape==(8,12)
    assert e.statistics(z)['mean_std']>0

def test_learning_updates_parameters_and_decreases_holdout_loss():
    torch.manual_seed(7)
    train=collect('key',11,80);held=collect('key',13,40)
    model=WorldModel(12,24,2,0.005)
    before=model.prediction_loss(held)
    original=model.encoder.net[0].weight.detach().clone()
    for i in range(80):
        ix=np.random.default_rng(i).integers(0,80,size=16)
        model.train_batch(tuple(x[ix] for x in train))
    after=model.prediction_loss(held)
    assert not torch.equal(original,model.encoder.net[0].weight)
    assert after < before, (before,after)

def test_invalid_shape():
    import pytest
    with pytest.raises(ValueError): ObservationEncoder()(torch.rand(1,3,9,9))
