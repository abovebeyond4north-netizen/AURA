"""No-oracle confidence learner tests and transactional failure preservation."""
import numpy as np
import pytest

from aura.environments.procedural_world import COLORS
from aura.exploration.door_probe import Unmeasurable
from aura.exploration.motion_confidence import (
    ConfidenceRollbackCamera, OnlineMotionConfidence, independent_visual_shift,
)


def frame(shift=(0,0), *, two=True):
    image = np.full((3,7,7), 127, dtype=np.uint8)
    anchors = {'switch': (2,3), 'crate': (4,2), 'goal': (5,5)}
    for i, (name, (r,c)) in enumerate(anchors.items()):
        if i == 1 and not two:
            break
        image[:,r-shift[0],c-shift[1]] = COLORS[name]
    image[:,3,3] = COLORS['agent']
    return image


def test_inference_abstains_without_two_agreeing_fixed_anchors():
    assert independent_visual_shift(frame(two=False),frame((0,1),two=False)) is None
    assert independent_visual_shift(frame(),frame()) is None
    corrupt = frame((0,1)); corrupt[:,2,2] = [127,127,127]; corrupt[:,2,4] = COLORS['switch']
    assert independent_visual_shift(frame(),corrupt) is None
    with pytest.raises(ValueError): independent_visual_shift(frame().astype('float64'),frame())


def test_learns_only_after_all_four_observed_directions_have_support():
    belief = OnlineMotionConfidence(min_evidence=6,threshold=.65)
    shifts=[(0,1),(1,0),(-1,0),(0,-1)]
    for raw,shift in enumerate(shifts):
        for _ in range(6):
            assert belief.observe_frames(raw,frame(),frame(shift))
        if raw < 3: assert belief.summary().committed_mapping is None
    result=belief.summary()
    assert result.committed_mapping == (3,1,0,2)
    assert result.observations==24
    assert all(x>=.65 for x in result.action_confidence)


def test_contradiction_rolls_back_all_mapping_confidence_without_oracle():
    belief=OnlineMotionConfidence(6,.65)
    shifts=[(-1,0),(1,0),(0,-1),(0,1)]
    for raw,shift in enumerate(shifts):
        for _ in range(6): belief.observe_frames(raw,frame(),frame(shift))
    assert belief.committed_mapping == (0,1,2,3)
    assert not belief.observe_frames(0,frame(),frame((1,0)))
    assert belief.summary().committed_mapping is None
    assert belief.summary().rollbacks == 1
    assert belief.summary().contradictions == 1
    assert belief.summary().action_support == (0,0,0,0)
    assert belief.audit_events[-1]['reason'] == 'observed movement contradicts committed mapping'
    # One surprising action cannot become a new accepted mapping.
    assert belief.summary().action_confidence == (0.,0.,0.,0.)


def test_invalid_raw_and_ambiguous_are_not_counted_as_mapping_evidence():
    b=OnlineMotionConfidence()
    with pytest.raises(ValueError): b.observe_frames(4,frame(),frame((1,0)))
    with pytest.raises(ValueError): b.observe_frames(True,frame(),frame((1,0)))
    assert not b.observe_frames(0,frame(two=False),frame((1,0),two=False))
    assert b.observations == 0 and b.abstentions == 1


def test_failed_reidentification_restores_complete_internal_state():
    im=frame()
    cam=ConfidenceRollbackCamera(im,lambda a:im.copy())
    original=(cam.mapping,cam.position,cam.image.copy(),cam.map.copy(),
              cam.landmarks.copy(),cam.change_resolutions)
    def fail_observe(image):
        cam.map[(777,777)]='tampered'
        cam.landmarks['door']=(777,777)
        raise Unmeasurable('injected inconsistent landmark')
    cam._observe=fail_observe
    with pytest.raises(Unmeasurable,match='injected'):
        cam._commit_reidentified((1,0,2,3),(1,0),im)
    assert cam.mapping==original[0]
    assert cam.position==original[1]
    assert np.array_equal(cam.image,original[2])
    assert cam.map==original[3]
    assert cam.landmarks==original[4]
    assert cam.change_resolutions==original[5]
    assert cam.failed_commits==1


def test_camera_no_privileged_state_access():
    import inspect
    source=inspect.getsource(ConfidenceRollbackCamera)+inspect.getsource(OnlineMotionConfidence)
    for forbidden in ('diagnostic_state','env.agent','env.door','oracle_mapping'):
        assert forbidden not in source

