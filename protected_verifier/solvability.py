"""Independent, privileged BFS *evaluation only*; never used for agent actions.

The evaluator may inspect hidden simulator state to certify task reachability.
This file must not be imported by the learner or training loop.
"""
from __future__ import annotations
from collections import deque
from copy import deepcopy

def evaluator_solution(env,max_nodes=5000):
    if not getattr(env,'_diagnostics',False):
        raise PermissionError('Only externally created diagnostic environments are certifiable')
    def key(w):
        return (w.agent,w.has_key,w.switch_on,w.door_open,w.crate,w._rule)
    seen={key(env)}
    queue=deque([(deepcopy(env),())])
    while queue and len(seen)<max_nodes:
        current,path=queue.popleft()
        if len(path)>=current.max_steps: continue
        for action in range(current.action_space.n):
            candidate=deepcopy(current)
            _,_,term,trunc,_=candidate.step(action)
            candidate_path=path+(action,)
            if term: return candidate_path
            state=key(candidate)
            if state not in seen and not trunc:
                seen.add(state);queue.append((candidate,candidate_path))
    return None
