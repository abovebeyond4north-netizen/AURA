"""Experimental partial-signature discrimination and novelty tie-break."""
from collections import defaultdict
from itertools import combinations
from types import FunctionType
from aura.exploration.entropy_probe import EntropyProbeCamera, select_entropy
from aura.exploration.hazard_dynamics import predict
from aura.exploration.persistent_pose import OFFSETS, STATIC
from aura.exploration.door_probe import offset, WALKABLE
from aura.environments.procedural_world import MOVES

def choose(branches, base, used):
    if len(branches)>128: return select_entropy(branches,base,used)
    ranks=[]
    for raw in range(4):
        if (frozenset(branches),raw) in used: continue
        groups=defaultdict(float); novelty=0.; moving=False
        for (m,p,mem),w in branches.items():
            tiles=base | dict(mem)
            if tiles.get(offset(p,MOVES[m[raw]])) not in WALKABLE | {'wall'}: break
            q,=predict((m,p),raw,tiles); moving |= q!=p
            sig=tuple(tiles.get(offset(q,d)) for d in OFFSETS)
            groups[q,sig]+=w
            novelty+=w*sum(t is None for t in sig)
        else:
            if not moving: continue
            total=sum(groups.values())
            if not total: continue
            gain=sum(a*b for ((p,s),a),((q,t),b) in combinations(groups.items(),2) if p!=q and any(x in STATIC and y in STATIC and x!=y for x,y in zip(s,t)))/total**2
            ranks.append((round(gain,12),round(novelty/total,12),-raw))
    return -max(ranks)[2] if ranks else None

class NoveltyCamera(EntropyProbeCamera):
    def act(self,action,expected):
        fn=EntropyProbeCamera.act
        return FunctionType(fn.__code__,dict(fn.__globals__,select_entropy=choose),closure=fn.__closure__)(self,action,expected)

if __name__=='__main__':
    import json,sys
    from pathlib import Path
    from experiments.probe_budget_ablation import trial
    rows=json.loads(Path(sys.argv[1]).read_text())
    with Path(sys.argv[2]).open('x') as f:
        for r in rows:
            if r['controller']!='EntropyProbeCamera' or r['size']!=15: continue
            for cls in (EntropyProbeCamera,NoveltyCamera):
                out=trial(r['seed']+960000,r['size'],r['column'],tuple(r['before']),tuple(r['after']),r['noise'],r['when'],cls,r['reset_start'])
                f.write(json.dumps(out)+'\n'); f.flush()
