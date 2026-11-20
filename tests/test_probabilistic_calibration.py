import numpy as np
import pytest
from aura.environments.procedural_world import COLORS
from aura.exploration.probabilistic_calibration import ProbabilisticCalibrationCamera
from aura.exploration.door_probe import Unmeasurable

def test_uninformative_calibration_does_not_commit():
    frame=np.tile(np.array(COLORS['empty'],dtype=np.uint8)[:,None,None],(1,7,7))
    frame[:,3,3]=COLORS['agent']
    cam=ProbabilisticCalibrationCamera(frame,lambda _:frame.copy())
    cam.map.update({(r,c):'empty' for r in range(-6,7) for c in range(-6,7)})
    before=cam.map.copy()
    with pytest.raises(Unmeasurable): cam.after_key_pickup()
    assert cam.map==before and cam.position==(0,0) and not cam.monitoring
