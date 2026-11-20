import numpy as np
from aura.world_model.dynamics import WorldModel
from aura.world_model.hierarchy import horizon_errors

def test_horizon_measurement_without_fabricating_long_data():
    model=WorldModel(8,16,2)
    obs=np.random.default_rng(8).integers(0,256,(5,3,7,7),dtype=np.uint8)
    result=horizon_errors(model,[(obs,[0,1,2,3])],(1,2,4))
    assert set(result)=={1,2,4} and all(np.isfinite(v) for v in result.values())
