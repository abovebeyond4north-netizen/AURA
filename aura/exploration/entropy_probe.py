"""Experimental pose information gain on jointly known static pixels."""
from collections import defaultdict
from math import log2
from aura.exploration.branch_probe import expand, advance
from aura.exploration.hypothesis_memory import HypothesisMemoryCamera
from aura.exploration.hazard_dynamics import predict
from aura.exploration.persistent_pose import view, OFFSETS, STATIC
from aura.exploration.door_probe import offset, Unmeasurable, WALKABLE
from aura.exploration.continuous_hazard import hazard_prior
from aura.environments.procedural_world import MOVES


def entropy(masses):
    total=sum(masses)
    return -sum((w/total)*log2(w/total) for w in masses if w>0) if total else 0.


def select_entropy(branches,base,used):
    best=None
    for raw in range(4):
        if (frozenset(branches),raw) in used: continue
        predictions=[]; moving=False
        for (m,p,mem),w in branches.items():
            tiles=base | dict(mem)
            if tiles.get(offset(p,MOVES[m[raw]])) not in WALKABLE | {'wall'}: break
            q,=predict((m,p),raw,tiles); moving |= q!=p
            predictions.append((q,w,tuple(tiles.get(offset(q,d)) for d in OFFSETS)))
        else:
            if not moving or not predictions: continue
            common=[i for i in range(len(OFFSETS)) if all(s[i] in STATIC for q,w,s in predictions)]
            groups=defaultdict(lambda: defaultdict(float)); poses=defaultdict(float)
            for q,w,s in predictions:
                groups[tuple(s[i] for i in common)][q]+=w; poses[q]+=w
            total=sum(poses.values())
            if not total: continue
            gain=entropy(list(poses.values()))-sum(sum(g.values())/total*entropy(list(g.values())) for g in groups.values())
            rank=(round(gain,12),-raw)
            if best is None or rank>best: best=rank
    return -best[1] if best is not None else None

class EntropyProbeCamera(HypothesisMemoryCamera):
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
            probe=select_entropy(predicted,base,used)
            if probe is None: break
            used.add((frozenset(predicted),probe)); old=image; self.change_probes+=1
            image=self.raw_step(probe)
            branches=advance(predicted,probe,view(old),view(image),base)
        raise Unmeasurable('continuous hazard pose unresolved')
