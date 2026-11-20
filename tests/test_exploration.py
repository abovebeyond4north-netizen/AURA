import numpy as np
from aura.exploration.curiosity import Curiosity,LearningProgress
from aura.world_model.dynamics import WorldModel

def test_fixed_sample_learning_progress():
    p=LearningProgress(window=3)
    for n in [.3,.29,.28]: assert p.update(n)==0
    assert p.update(.20)>0

def test_noisy_tv_is_penalized():
    c=Curiosity(WorldModel(8,16,2),noise_penalty=10)
    low=c.score(epistemic=1,action=0,noise_estimate=.01)
    noisy=c.score(epistemic=1,action=0,noise_estimate=100)
    assert low>noisy
    # Tests this scoring rule, not a full benchmark demonstration.

def test_curiosity_finite_action():
    c=Curiosity(WorldModel(8,16,2))
    obs=np.zeros((3,7,7),dtype=np.uint8)
    assert c.choose(obs,rng=np.random.default_rng(2)) in range(5)
