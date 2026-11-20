"""Instrument the unchanged v74 controller; evaluator traces never enter policy."""
import argparse,json
from pathlib import Path
from collections import Counter
from unittest.mock import patch
from aura.exploration import continuous_hazard as ch
from aura.exploration.frontier_approach import FrontierApproachCamera
from experiments.door_route_recovery_study import trial


def diagnose(rows):
    results=[]
    for row in rows:
        if row['controller']!='FrontierApproachCamera': continue
        events=[]
        original_update=ch.update
        original_probe=ch.qualified_probe
        def update(weights,raw,old,new,tiles):
            out=original_update(weights,raw,old,new,tiles)
            events.append({'event':'update','raw':raw,'prior':len(weights),'posterior':len(out),'poses':len({p for m,p in out})})
            return out
        def probe(weights,tiles,used):
            out=original_probe(weights,tiles,used)
            events.append({'event':'probe','support':len(weights),'action':out})
            return out
        with patch.object(ch,'update',update),patch.object(ch,'qualified_probe',probe):
            actual=trial(row['seed'],row['size'],row['column'],tuple(row['before']),tuple(row['after']),row['noise'],row['when'],FrontierApproachCamera,row['reset_start'])
        matches=json.loads(json.dumps(actual))==row
        if not matches: raise RuntimeError('instrumentation changed baseline')
        last=events[-1] if events else {}
        category='success' if actual['success'] else ('empty posterior' if last.get('posterior')==0 else 'no qualified probe' if last.get('event')=='probe' and last.get('action') is None else 'other')
        results.append({'trial':actual,'events':events,'category':category,'exact_replay':matches})
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    results=diagnose([r for f in a.inputs for r in json.loads(f.read_text())])
    with a.output.open('x') as f: json.dump(results,f,indent=2)
    print(Counter(r['category'] for r in results))
