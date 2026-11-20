import json
from pathlib import Path
import pytest
from aura.exploration.goal_search import RecoveryGoalCamera, PassageRecoveryCamera
from aura.exploration.door_probe import Unmeasurable
from experiments.goal_acquisition import trial


def test_occluded_passages():
    for name in ('passage_dev.json','passage_fresh.json'):
        rows=json.loads((Path(__file__).parents[1]/'verification/section08'/name).read_text())
        for row in rows:
            if row['arm']=='baseline' and row['reason']=='door surface occluded':
                args=[row[k] for k in ('seed','size','column','reverse','noise','policy')]
                assert trial(*args,RecoveryGoalCamera)['reason']=='door surface occluded'
                assert trial(*args,PassageRecoveryCamera)['success']


def test_missing_surface_does_not_bypass_motion_verification():
    camera=object.__new__(PassageRecoveryCamera)
    camera.prepare=lambda arm: 0
    camera.at_pixel=lambda delta: None
    def act(action, expected):
        if action != 4:
            raise Unmeasurable('movement inconsistent with camera contract')
    camera.act=act
    with pytest.raises(Unmeasurable):
        camera.measure(2)
