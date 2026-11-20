import numpy as np
import pytest
from aura.environments.procedural_world import COLORS
from aura.exploration.door_probe import Unmeasurable
from aura.exploration.learned_dynamics import LearnedDynamicsCamera
from aura.exploration.context_dynamics import ContextDynamicsCamera
from experiments.context_dynamics_study import trial


def test_unresolved_context_does_not_commit_map_or_mapping():
    frame = np.tile(np.array(COLORS['empty'],dtype=np.uint8)[:,None,None],(1,7,7))
    frame[:,3,3] = COLORS['agent']
    camera = ContextDynamicsCamera(frame,lambda a:frame.copy())
    camera.mapping = (2,0,3,1)
    camera.map.update({(r,c):'empty' for r in range(-5,6) for c in range(-5,6)})
    before = camera.map.copy()
    with pytest.raises(Unmeasurable,match='post-passage action mapping unresolved'):
        camera.after_verified_passage()
    assert camera.map == before
    assert camera.mapping == (2,0,3,1)
    assert camera.position == (0,0)
    assert camera.context_calibrations == 0
    assert camera.context_probes <= camera.max_calibration_probes


@pytest.mark.parametrize('after',[(3,2,1,0),(2,0,3,1),(0,1,2,3)])
def test_context_mapping_reidentification(after):
    row = trial(96000,9,5,(0,1,2,3),after,0,ContextDynamicsCamera)
    assert row['success'] and row['passed']
    assert not row['pose_errors'] and row['mapping_errors']==0
    assert len(row['learned'])==2 and row['learned'][1][1]==after
    assert 0 < row['context_probes'] <= 12
    assert len(row['actions'])<=65


def test_control_cannot_reidentify_changed_mapping():
    row = trial(96000,9,5,(0,1,2,3),(3,2,1,0),0,LearnedDynamicsCamera)
    assert row['passed'] and not row['success']
    assert len(row['learned'])==1 and row['context_probes']==0
