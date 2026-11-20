import json,sys
from pathlib import Path
from aura.exploration.hypothesis_memory import HypothesisMemoryCamera
from experiments.door_route_recovery_study import trial

def run(paths):
    return [trial(r['seed'],r['size'],r['column'],tuple(r['before']),tuple(r['after']),r['noise'],r['when'],HypothesisMemoryCamera,r['reset_start']) for p in paths for r in json.loads(Path(p).read_text()) if r['controller']=='FrontierApproachCamera']
if __name__=='__main__':
    with open(sys.argv[1],'x') as f: json.dump(run(sys.argv[2:]),f,indent=2)
