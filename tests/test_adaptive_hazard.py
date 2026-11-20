import pytest
from aura.exploration.adaptive_hazard import ResetRate, AdaptiveHazardCamera
from aura.exploration.learned_dynamics import PERMUTATIONS
from experiments.change_dynamics_study import trial

def test_no_information_preserves_rate():
    r=ResetRate(); h=r.mean; prior={PERMUTATIONS[0]:1.}
    posterior={m:(1-h)*prior.get(m,0)+h/24 for m in PERMUTATIONS}
    r.observe(prior,posterior,h)
    assert r.mean==pytest.approx(h)

def test_confirmed_change_increases_rate_and_stability_decreases_it():
    r=ResetRate(); p={PERMUTATIONS[0]:1.}; h=r.mean
    r.observe(p,{PERMUTATIONS[1]:1.},h); assert r.mean>h
    h=r.mean; r.observe(p,p,h); assert 0<r.mean<h

def test_camera_fixture():
    row=trial(97000,9,5,PERMUTATIONS[0],PERMUTATIONS[11],0,'late',AdaptiveHazardCamera)
    assert not row['pose_errors'] and not row['mapping_errors']
