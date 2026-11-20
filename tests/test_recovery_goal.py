from aura.exploration.goal_search import PaletteGoalCamera, RecoveryGoalCamera
from experiments.goal_acquisition import trial
import json
from pathlib import Path

def test_recovered_calibration():
    rows=json.loads((Path(__file__).parents[1]/'verification/section08/recovery_fresh.json').read_text())
    for reason in ('no clear calibration axis','ambiguous calibration motion'):
        old=next(r for r in rows if r['arm']=='baseline' and r['reason']==reason)
        args=[old[k] for k in ('seed','size','column','reverse','noise','policy')]
        assert trial(*args,PaletteGoalCamera)['reason']==reason
        assert trial(*args,RecoveryGoalCamera)['success']

def test_clean_controls():
    for reverse in (False,True):
        assert trial(74100,11,4,reverse,0,'frontier',RecoveryGoalCamera)['success']
