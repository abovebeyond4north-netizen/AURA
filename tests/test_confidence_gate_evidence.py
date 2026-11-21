import copy
import pytest
from experiments.confidence_gate_study import CONTROLLERS, SEEDS, LAYOUTS, NOISE, summarize


def _rows():
    return [{'seed':seed, 'size':size, 'column':col, 'reverse':reverse,
             'noise':noise, 'controller':cls.__name__, 'success':False,
             'actions':[0,2,4], 'pose_errors':[]}
            for seed in SEEDS for size,col in LAYOUTS
            for reverse in (False,True) for noise in NOISE for cls in CONTROLLERS]


def test_complete_public_matrix():
    s = summarize(_rows())
    assert all(x['trials']==72 and x['successes']==0 for x in s.values())


def test_rejects_missing_unexpected_corrupt_trials():
    good = _rows()
    for mutate in (lambda rows: rows.pop(),
                   lambda rows: rows.append(copy.deepcopy(rows[0])),
                   lambda rows: rows[0].update(seed=111),
                   lambda rows: rows[0].update(actions=[7]),
                   lambda rows: rows[0].update(success=1),
                   lambda rows: rows[0].update(pose_errors=[{'predicted':[1,1]}])):
        candidate = copy.deepcopy(good)
        mutate(candidate)
        with pytest.raises((ValueError,KeyError,TypeError)):
            summarize(candidate)
