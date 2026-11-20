"""Bounded branch-local static evidence; never merge ambiguous maps."""
from collections import defaultdict
from math import fsum
from aura.exploration.continuous_hazard import hazard_prior
from aura.exploration.learned_dynamics import PERMUTATIONS
from aura.exploration.hazard_dynamics import predict, normalize, qualified_probe
from aura.exploration.persistent_pose import compatible, view
from aura.exploration.door_probe import offset, Unmeasurable
from aura.exploration.frontier_approach import FrontierApproachCamera


def advance(branches, raw, old, new, base, hazard=0.):
    result=defaultdict(float)
    for (m,p,memory),weight in branches.items():
        tiles=base | dict(memory)
        for mapping in PERMUTATIONS if hazard else (m,):
            w=weight*((1-hazard)*(mapping==m)+hazard/len(PERMUTATIONS))
            outcomes=predict((mapping,p),raw,tiles)
            for q in outcomes:
                if not compatible(old,new,p,q,tiles): continue
                extra=dict(memory)
                for d,t in new.items():
                    cell=offset(q,d)
                    if cell not in base: extra[cell]=t
                result[mapping,q,tuple(sorted(extra.items()))]+=w/len(outcomes)
    return normalize(result)

class HypothesisMemoryCamera(FrontierApproachCamera):
    def act(self,action,expected):
        if not self.monitoring or action>=4: return super().act(action,expected)
        target=offset(self.position,expected); base=self.map.copy(); old=self.image
        raw=self.mapping.index(action); image=self.raw_step(raw)
        prior=hazard_prior(self.mapping_belief or {self.mapping:1.},self.position,self.hazard)
        branches=advance({(m,p,()):w for (m,p),w in prior.items()},raw,view(old),view(image),base)
        used=set()
        for n in range(self.max_calibration_probes+1):
            if not branches or len(branches)>4096: break
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
            from aura.exploration.continuous_hazard import propagate
            predicted=propagate(weights,self.hazard)
            if len(predicted)>self.max_calibration_support: break
            probe=qualified_probe(predicted,base,used)
            if probe is None: break
            used.add((frozenset(predicted),probe)); old=image; self.change_probes+=1
            image=self.raw_step(probe)
            branches=advance(branches,probe,view(old),view(image),base,self.hazard)
        raise Unmeasurable('continuous hazard pose unresolved')
