import numpy as np
import pytest
from aura.exploration.pose_belief import PoseBelief


def frame(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(3,7,7), dtype=np.uint8)


def translated(before, shift):
    dy, dx = shift
    after = np.zeros_like(before)
    for r in range(7):
        for c in range(7):
            rr, cc = r + dy, c + dx
            if 0 <= rr < 7 and 0 <= cc < 7:
                after[:, r, c] = before[:, rr, cc]
    return after


@pytest.mark.parametrize('shift', [(0,0),(-1,0),(1,0),(0,-1),(0,1)])
def test_shift_recovery(shift):
    b = frame(44)
    estimate = PoseBelief().update(b, translated(b,shift), expected=shift)
    assert estimate.shift == shift
    assert estimate.confidence >= .995


def test_noisy_shift_recovery():
    b = frame(13)
    a = translated(b, (0,1))
    a[:, 0, 0] = (17,18,19)
    assert PoseBelief().update(b,a,expected=(0,1)).shift == (0,1)


def test_ambiguous_uniform_abstains():
    b = np.zeros((3,7,7),dtype=np.uint8)
    assert PoseBelief(threshold=.995).update(b,b).shift is None


def test_invalid_parameters():
    with pytest.raises(ValueError): PoseBelief(corruption=0)
    with pytest.raises(ValueError): PoseBelief().update(frame(),frame(),expected=(2,0))
