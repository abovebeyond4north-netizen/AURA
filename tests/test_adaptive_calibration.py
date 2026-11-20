import numpy as np
import pytest
from aura.environments.procedural_world import COLORS
from aura.exploration.adaptive_calibration import advance, choose_raw, CalibratedComposedCamera
from aura.exploration.door_probe import Unmeasurable
from experiments.adaptive_calibration_study import audited_trial


def test_uninformative_frames_preserve_both_mappings():
    assert advance({(False,(0,0)),(True,(0,0))},2,{},{},{}) == {(False,(0,-1)),(True,(0,1)),(False,(0,0)),(True,(0,0))}


def test_static_evidence_resolves_mapping_and_pose():
    tiles={(0,-1):'wall',(0,1):'empty',(0,2):'wall',(0,3):'empty'}
    assert advance({(False,(0,0)),(True,(0,0))},2,{}, {(0,1):'wall',(0,2):'empty'},tiles)=={(True,(0,1))}


def test_probe_rejects_unknown_and_dynamic_targets():
    states={(False,(0,0)),(True,(0,0))}
    assert choose_raw(states,{},set()) is None
    assert choose_raw(states,{(0,1):'empty',(0,-1):'door'},set()) is None
    assert choose_raw(states,{(0,1):'empty',(0,-1):'wall'},set()) in (2,3)


def test_uninformative_probe_does_not_commit():
    frame=np.tile(np.array(COLORS['empty'],dtype=np.uint8)[:,None,None],(1,7,7))
    frame[:,3,3]=COLORS['agent']
    c=CalibratedComposedCamera(frame,lambda a:frame.copy())
    c.map.update({(r,k):'empty' for r in range(-7,8) for k in range(-7,8)})
    before=c.map.copy()
    with pytest.raises(Unmeasurable,match='joint calibration unresolved'):
        c.after_key_pickup()
    assert c.map==before and c.position==(0,0) and not c.reversed
    assert 0<c.probes<=c.max_calibration_probes


@pytest.mark.parametrize('rev',[False,True])
def test_real_mapping_and_pose(rev):
    row=audited_trial(93200,9,6,rev,.08,CalibratedComposedCamera)
    assert row['completed_calibrations']==1 and row['mapping_errors']==0
    assert not row['pose_errors']
