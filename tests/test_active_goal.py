import numpy as np
import pytest
from aura.exploration.goal_search import ActiveRecoveryCamera
from aura.exploration.door_probe import Unmeasurable
from aura.environments.procedural_world import COLORS


def test_recovery_abstains_when_no_shared_wall():
    frame = np.broadcast_to(np.array(COLORS['empty'],dtype=np.uint8)[:,None,None],(3,7,7)).copy()
    calls=[]
    c=ActiveRecoveryCamera(frame,lambda a: (calls.append(a),frame)[1])
    with pytest.raises(Unmeasurable): c.act(0,(-1,0))
    assert c.position==(0,0) and len(calls)==1
