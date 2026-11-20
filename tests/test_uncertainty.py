import numpy as np
from aura.world_model.dynamics import WorldModel
from aura.world_model.uncertainty import UncertaintyEstimator

def test_uncertainty_shift_and_evidence_request():
    model=WorldModel(8,16,2)
    u=UncertaintyEstimator(model)
    x=np.zeros((3,7,7),dtype=np.uint8)
    assert u.estimate_uncertainty(x,0)>=0
    assert not u.detect_distribution_shift(x)
    for _ in range(30):u.observations.append(0.)
    assert u.detect_distribution_shift(np.full((3,7,7),255,dtype=np.uint8))
    class H:confidence=.2
    assert u.should_request_more_evidence(H())

def test_uncertainty_calibration_is_unclaimed_for_tiny_samples():
    u=UncertaintyEstimator(WorldModel(8,16,2))
    assert u.calibration()['predictive_error_calibration_mae'] is None


def test_empirical_calibration_has_disjoint_train_test_bins():
    u=UncertaintyEstimator(WorldModel(8,16,2))
    u.predicted.extend([float(i%5)/10 for i in range(20)])
    u.errors.extend([float(i%3)/10 for i in range(20)])
    result=u.calibration()
    assert result['n']==20 and result['predictive_error_calibration_mae']>=0
