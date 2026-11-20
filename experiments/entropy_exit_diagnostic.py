"""Observe original controller exit lines without altering its decisions."""
import json,sys
from collections import Counter,deque
from pathlib import Path
from aura.exploration.entropy_probe import EntropyProbeCamera
from experiments.door_route_recovery_study import trial

def run(rows):
    out=[]
    code=EntropyProbeCamera.act.__code__
    for row in rows:
        events=deque(maxlen=12); last=[]
        def trace(frame,event,arg):
            nonlocal last
            if frame.f_code is not code: return None
            if event=='call': events.clear()
            if event in ('line','exception'):
                loc=frame.f_locals
                events.append(dict(event=event,line=frame.f_lineno,probe=loc.get('probe'),iteration=loc.get('n'),branches=len(loc['branches']) if 'branches' in loc else None,poses=len(loc['poses']) if 'poses' in loc else None))
                if event=='exception': last=list(events)
            return trace
        previous=sys.gettrace()
        try:
            sys.settrace(trace)
            actual=trial(row['seed'],row['size'],row['column'],tuple(row['before']),tuple(row['after']),row['noise'],row['when'],EntropyProbeCamera,row['reset_start'])
        finally: sys.settrace(previous)
        if json.loads(json.dumps(actual))!=row: raise RuntimeError('replay mismatch')
        out.append(dict(trial=actual,terminal_trace=last))
    return out
if __name__=='__main__':
    rows=json.loads(Path(sys.argv[1]).read_text())
    result=run(rows)
    with open(sys.argv[2],'x') as f: json.dump(result,f,indent=2)
    print(Counter(tuple(e['line'] for e in r['terminal_trace'][-4:]) for r in result if not r['trial']['success']))
