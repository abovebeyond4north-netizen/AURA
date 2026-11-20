"""Replay scale failures without changing controller decisions."""
import json, sys
from pathlib import Path
from collections import deque
from aura.exploration.entropy_probe import EntropyProbeCamera
from experiments.probe_budget_ablation import Budget6, trial

def diagnose(row):
    history=deque(maxlen=16)
    terminal=[]
    code=EntropyProbeCamera.act.__code__
    def trace(frame,event,arg):
        if frame.f_code is not code: return None
        if event=='call': history.clear()
        if event in ('line','exception'):
            loc=frame.f_locals
            branches=loc.get('branches',{})
            history.append(dict(line=frame.f_lineno,event=event,n=loc.get('n'),probe=loc.get('probe'),branches=len(branches),poses=len({p for _,p,_ in branches}),support=len(loc.get('predicted',{}))))
            if event=='exception' and arg[0].__name__=='Unmeasurable': terminal[:] = list(history)
        return trace
    cls={'Budget6':Budget6,'EntropyProbeCamera':EntropyProbeCamera}[row['controller']]
    old=sys.gettrace()
    try:
        sys.settrace(trace)
        actual=trial(row['seed'],row['size'],row['column'],tuple(row['before']),tuple(row['after']),row['noise'],row['when'],cls,row['reset_start'])
    finally: sys.settrace(old)
    return dict(expected=row,actual=actual,exact=json.loads(json.dumps(actual))==row,terminal=terminal)

if __name__=='__main__':
    rows=json.loads(Path(sys.argv[1]).read_text())
    with Path(sys.argv[2]).open('x') as f:
        for row in rows:
            if row['size']==15:
                f.write(json.dumps(diagnose(row))+'\n'); f.flush()
