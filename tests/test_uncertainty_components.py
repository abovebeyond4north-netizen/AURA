"""Tests for distinct uncertainty observables and planner conservatism."""
import numpy as np
import torch
import pytest
from aura.world_model.uncertainty import UncertaintyEstimator
from aura.world_model.dynamics import WorldModel
from aura.planning.planner import Planner


def test_conditional_variance_distinguishes_repeatability_from_stochasticity():
    u=UncertaintyEstimator(WorldModel(8,16,2))
    current=np.zeros((3,7,7),dtype=np.uint8)
    assert u.conditional_outcome_variance(current,2) is None
    for _ in range(12):
        u.record_observed_outcome(current,2,current)
    assert u.conditional_outcome_variance(current,2)==0.0
    # Another action, same observed state, has independent stochastic outcomes.
    for i in range(12):
        target=np.full((3,7,7),255 if i%2 else 0,dtype=np.uint8)
        u.record_observed_outcome(current,3,target)
    assert u.conditional_outcome_variance(current,3)>.2
    assert u.conditional_outcome_variance(current,4) is None
    with pytest.raises(ValueError):
        u.record_observed_outcome(current,7,current)


def test_frozen_ood_detector_uses_calibration_data_only():
    u=UncertaintyEstimator(WorldModel(8,16,2))
    rng=np.random.default_rng(50)
    data=[rng.integers(0,12,size=(3,7,7),dtype=np.uint8) for _ in range(80)]
    calibrated=u.fit_shift_detector(data,alpha=.05)
    assert calibrated['reference_size']==40
    assert calibrated['calibration_size']==40
    normal=rng.integers(0,12,size=(3,7,7),dtype=np.uint8)
    shifted=np.zeros((3,7,7),dtype=np.uint8);shifted[0]=255
    assert not u.detect_distribution_shift(normal)
    assert u.detect_distribution_shift(shifted)
    assert u.shift_report(shifted)['method']=='frozen_empirical_nearest_reference'
    with pytest.raises(ValueError):
        u.fit_shift_detector(data[:10])
    with pytest.raises(ValueError):
        u.fit_shift_detector(data,alpha=.75)


def test_ensemble_and_observation_noise_not_conflated():
    u=UncertaintyEstimator(WorldModel(8,16,2))
    x=np.zeros((3,7,7),dtype=np.uint8)
    before=u.estimate_components(x,0)
    assert before['ensemble_disagreement']>=0
    assert before['observed_conditional_variance'] is None
    for i in range(6):
        u.record_observed_outcome(x,0,np.full_like(x,255 if i%2 else 0))
    after=u.estimate_components(x,0)
    assert after['observed_conditional_variance']>.2
    assert np.isclose(after['ensemble_disagreement'],before['ensemble_disagreement'])


def test_planner_shortens_on_shift_and_chooses_information_action():
    class World:
        def estimate_uncertainty(self,x,actions):
            return torch.zeros(len(actions),dtype=torch.float32)
    class Actor:
        def select(self,*args):return 1
    class Curiosity:
        def choose(self,*args,**kwargs):return 4
    class Shift:
        shift=False
        def detect_distribution_shift(self,obs):return self.shift
    shift=Shift()
    planner=Planner(World(),Actor(),Curiosity(),uncertainty_limit=.5,uncertainty_estimator=shift)
    obs=np.zeros((3,7,7),dtype=np.uint8)
    assert planner.choose(obs,np.random.default_rng(1),horizon=4)==(1,4)
    shift.shift=True
    assert planner.choose(obs,np.random.default_rng(1),horizon=4)==(4,1)


def test_empirical_risk_bound_reports_measured_not_guaranteed_coverage():
    u=UncertaintyEstimator(WorldModel(8,16,2))
    u.predicted.extend(np.linspace(0,1,80).tolist())
    u.errors.extend((.05+np.linspace(0,.1,80)).tolist())
    calibration=u.calibration(alpha=.1)
    assert 0<=calibration['empirical_error_bound_coverage']<=1
    assert calibration['requested_upper_bound_coverage']==.9
    assert u.prediction_error_bound() is not None
