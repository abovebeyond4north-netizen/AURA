"""Experimental branch-specific probe selection with fail-closed expansion limits."""
from collections import defaultdict
from itertools import combinations
from aura.exploration.hypothesis_memory import HypothesisMemoryCamera
from aura.exploration.hazard_dynamics import predict, normalize
from aura.exploration.persistent_pose import compatible, view, OFFSETS, STATIC
from aura.exploration.door_probe import offset, Unmeasurable, WALKABLE
from aura.exploration.learned_dynamics import PERMUTATIONS
from aura.exploration.continuous_hazard import hazard_prior
from aura.environments.procedural_world import MOVES

def expand(branches, hazard, cap=4096):
    result=defaultdict(float)
    for (m,p,mem), weight in branches.items():
        for mapping in PERMUTATIONS if hazard else (m,):
            k=(mapping,p,mem)
            if k not in result and len(result)>=cap: raise Unmeasurable('branch budget exceeded')
            result[k]+=weight*((1-hazard)*(mapping==m)+hazard/len(PERMUTATIONS))
    return normalize(result)

def advance(branches,raw,old,new,base,cap=4096):
    result=defaultdict(float)
    for (m,p,mem),w in branches.items():
        tiles=base | dict(mem); outcomes=predict((m,p),raw,tiles)
        for q in outcomes:
            if not compatible(old,new,p,q,tiles): continue
            extra=dict(mem)
            for d,t in new.items():
                cell=offset(q,d)
                if cell not in base: extra[cell]=t
            k=(m,q,tuple(sorted(extra.items())))
            if k not in result and len(result)>=cap: raise Unmeasurable('branch budget exceeded')
            result[k]+=w/len(outcomes)
    return normalize(result)

def select(branches,base,used):
    best=None
    for raw in range(4):
        if (frozenset(branches),raw) in used: continue
        signatures=defaultdict(float); moving=False
        for (m,p,mem),w in branches.items():
            tiles=base | dict(mem); target=offset(p,MOVES[m[raw]])
            if tiles.get(target) not in WALKABLE | {'wall'}: break
            q,=predict((m,p),raw,tiles); moving |= q!=p
            sig=tuple(tiles.get(offset(q,d)) for d in OFFSETS)
            signatures[sig]+=w
        else:
            if not moving: continue
            score=sum(a*b*min(3,sum(x in STATIC and y in STATIC and x!=y for x,y in zip(s,t))) for (s,a),(t,b) in combinations(signatures.items(),2))
            rank=(score,-raw)
            if best is None or rank>best: best=rank
    return -best[1] if best is not None else None

class BranchProbeCamera(HypothesisMemoryCamera):
    def act(self,action,expected):
        if not self.monitoring or action>=4: return super().act(action,expected)
        target=offset(self.position,expected); base=self.map.copy(); old=self.image
        raw=self.mapping.index(action); image=self.raw_step(raw)
        prior=hazard_prior(self.mapping_belief or {self.mapping:1.},self.position,self.hazard)
        branches=advance({(m,p,()):w for (m,p),w in prior.items()},raw,view(old),view(image),base)
        used=set()
        for n in range(self.max_calibration_probes+1):
            if not branches: break
            weights=defaultdict(float)
            for (m,p,_),w in branches.items(): weights[m,p]+=w
            poses={p for m,p in weights}
            if len(poses)==1:
                p=next(iter(poses))
                if sum(base.get(offset(p,d))==t for d,t in view(image).items())>=12:
                    self._marginal(weights); self.mapping=max(sorted(self.mapping_belief),key=self.mapping_belief.get)
                    self.position=p; self._observe(image)
                    if n: self.pose_only_resolutions+=1
                    if p!=target: self.recover_target(target)
                    return
            if n==self.max_calibration_probes: break
            if n==0: self.change_events+=1
            predicted=expand(branches,self.hazard)
            if len({(m,p) for m,p,_ in predicted})>self.max_calibration_support: break
            probe=select(predicted,base,used)
            if probe is None: break
            used.add((frozenset(predicted),probe)); old=image; self.change_probes+=1
            image=self.raw_step(probe)
            branches=advance(predicted,probe,view(old),view(image),base)
        raise Unmeasurable('continuous hazard pose unresolved')
