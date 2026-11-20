"""Public diagnostic: vary per-event probe cap, keeping environment cap fixed."""
import json, sys
from pathlib import Path
from aura.exploration.entropy_probe import EntropyProbeCamera
from experiments.door_route_recovery_study import trial
class Budget6(EntropyProbeCamera): max_calibration_probes = 6
class Budget24(EntropyProbeCamera): max_calibration_probes = 24
def run(rows):
    return [trial(r['seed'],r['size'],r['column'],tuple(r['before']),tuple(r['after']),r['noise'],r['when'],cls,r['reset_start']) for cls in (EntropyProbeCamera,Budget6,Budget24) for r in rows]
if __name__=='__main__':
    rows=json.loads(Path(sys.argv[1]).read_text())
    with open(sys.argv[2],'x') as f: json.dump(run(rows),f,indent=2)
