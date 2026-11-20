"""Opt-in two-step entropy ranking; no promotion or verifier changes."""
from collections import defaultdict
from aura.exploration.entropy_probe import entropy, select_entropy, EntropyProbeCamera
from aura.exploration.branch_probe import expand
from aura.exploration.hazard_dynamics import predict
from aura.exploration.persistent_pose import OFFSETS, STATIC
from aura.exploration.door_probe import offset, WALKABLE
from aura.environments.procedural_world import MOVES

def partition(branches, base, raw):
    rows=[]
    for (m,p,mem),w in branches.items():
        tiles=base | dict(mem)
        if tiles.get(offset(p,MOVES[m[raw]])) not in WALKABLE | {'wall'}: return None
        q,=predict((m,p),raw,tiles)
        rows.append(((m,q,mem),w,tuple(tiles.get(offset(q,d)) for d in OFFSETS),q!=p))
    if not rows or not any(r[3] for r in rows): return None
    common=[i for i in range(len(OFFSETS)) if all(r[2][i] in STATIC for r in rows)]
    groups=defaultdict(lambda:defaultdict(float))
    for k,w,s,_ in rows: groups[tuple(s[i] for i in common)][k]+=w
    return groups

def residual(groups):
    total=sum(sum(g.values()) for g in groups.values())
    result=0.
    for g in groups.values():
        poses=defaultdict(float)
        for (_,p,_),w in g.items(): poses[p]+=w
        result+=sum(poses.values())/total*entropy(list(poses.values()))
    return result

def choose(branches, base, used, hazard):
    if len(branches)>128: return select_entropy(branches,base,used)
    ranks=[]
    for raw in range(4):
        if (frozenset(branches),raw) in used: continue
        groups=partition(branches,base,raw)
        if not groups: continue
        immediate=residual(groups); future=0.; total=sum(sum(g.values()) for g in groups.values())
        for g in groups.values():
            mass=sum(g.values()); current=residual({0:g})
            if current==0: continue
            if hazard and len(g)*24>128: future+=mass/total*current; continue
            nxt=expand(g,hazard)
            scores=[residual(v) for a in range(4) if (v:=partition(nxt,base,a))]
            future+=mass/total*min([current]+scores)
        ranks.append((round(immediate+.5*future,12),raw))
    return min(ranks)[1] if ranks else None

class TwoStepCamera(EntropyProbeCamera):
    def act(self,action,expected):
        # Reuse the exact baseline loop with an isolated function namespace.
        from types import FunctionType
        fn=EntropyProbeCamera.act
        scope=dict(fn.__globals__,select_entropy=lambda b,m,u:choose(b,m,u,self.hazard))
        return FunctionType(fn.__code__,scope,closure=fn.__closure__)(self,action,expected)

if __name__=='__main__':
    import json,sys
    from pathlib import Path
    from experiments.probe_budget_ablation import trial
    rows=json.loads(Path(sys.argv[1]).read_text())
    with Path(sys.argv[2]).open('x') as f:
        for r in rows:
            if r['controller']!='EntropyProbeCamera' or r['size']!=15: continue
            out=trial(r['seed'],r['size'],r['column'],tuple(r['before']),tuple(r['after']),r['noise'],r['when'],TwoStepCamera,r['reset_start'])
            f.write(json.dumps({'baseline':r,'candidate':out})+'\n'); f.flush()
