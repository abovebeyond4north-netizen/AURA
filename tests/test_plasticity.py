import torch
import numpy as np
from aura.world_model.dynamics import WorldModel
from aura.meta_learning.plasticity import PlasticityMonitor

def test_plasticity_rejuvenation_is_isolated():
    model=WorldModel(8,16,2)
    original=[p.detach().clone() for p in model.parameters()]
    observations=np.zeros((10,3,7,7),dtype=np.uint8)
    monitor=PlasticityMonitor(.02)
    candidate,info=monitor.candidate_rejuvenation(model,observations,.25)
    assert info['requires_independent_promotion']
    assert 0<=len(info['rejuvenated_indices'])<=2
    assert all(torch.equal(p,q) for p,q in zip(original,model.parameters()))
    assert any(not torch.equal(p,q) for p,q in zip(original,candidate.parameters()))
