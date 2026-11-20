import json
from pathlib import Path
from collections import Counter
from experiments.localization_diagnostic import diagnose

ROOT=Path(__file__).resolve().parents[1]

def test_diagnostic_replays_success_and_failure():
    rows=json.loads((ROOT/'verification/section08/v76/development.json').read_text())
    baseline=[r for r in rows if r['controller']=='FrontierApproachCamera']
    chosen=[next(r for r in baseline if r['success']==v) for v in (True,False)]
    result=diagnose(chosen)
    assert all(r['exact_replay'] for r in result)
    assert result[0]['category']=='success'
    assert result[1]['category']!='success'
    assert result[1]['events'][-1]
