"""Hazard-weighted mapping beliefs; hard visual certificates for commitments.
The hazard is a fixed prior, not an estimated change rate. No probability pruning.
"""
from collections import defaultdict
from math import fsum
from aura.environments.procedural_world import MOVES
from aura.exploration.door_probe import WALKABLE, Unmeasurable, offset
from aura.exploration.change_dynamics import ReplanningChangeCamera
from aura.exploration.learned_dynamics import PERMUTATIONS
from aura.exploration.persistent_pose import OFFSETS, STATIC, compatible, view


def normalize(weights):
    total = fsum(weights.values())
    return {k: v/total for k,v in weights.items()} if total > 0 else {}


def hazard_prior(belief, origin, hazard):
    if not 0 < hazard <= 1:
        raise ValueError('hazard must be in (0,1]')
    return {(m,origin): (1-hazard)*belief.get(m,0)+hazard/len(PERMUTATIONS) for m in PERMUTATIONS}


def predict(state, raw, tiles):
    mapping,p=state
    q=offset(p,MOVES[mapping[raw]])
    tile=tiles.get(q)
    return (p,) if tile=='wall' else (q,) if tile in WALKABLE | {'goal'} else (p,q)


def update(weights,raw,old,new,tiles):
    nxt=defaultdict(float)
    for state,w in sorted(weights.items()):
        mapping,p=state
        outcomes=predict(state,raw,tiles)
        for q in outcomes:
            if compatible(old,new,p,q,tiles): nxt[mapping,q]+=w/len(outcomes)
    return normalize(nxt)


def qualified_probe(weights,tiles,used):
    ranked=[]
    for raw in range(4):
        if (frozenset(weights),raw) in used: continue
        if any(tiles.get(offset(p,MOVES[m[raw]])) not in WALKABLE | {'wall'} for m,p in weights): continue
        mass=defaultdict(float)
        for state,w in sorted(weights.items()):
            q,=predict(state,raw,tiles)
            mass[q]+=w
        if all(p==predict((m,p),raw,tiles)[0] for m,p in weights): continue
        pairs=sorted(mass)
        score=0.
        for i,p in enumerate(pairs):
            for q in pairs[i+1:]:
                contrast=sum(tiles.get(offset(p,d)) in STATIC and tiles.get(offset(q,d)) in STATIC and tiles[offset(p,d)]!=tiles[offset(q,d)] for d in OFFSETS)
                score+=mass[p]*mass[q]*min(contrast,3)
        ranked.append((score,len(mass),-raw))
    return -max(ranked)[2] if ranked else None


class HazardDynamicsCamera(ReplanningChangeCamera):
    hazard=.03
    def __init__(self,image,step):
        self.mapping_belief={}
        self.pose_only_resolutions=0
        self.posterior_updates=0
        super().__init__(image,step)

    def after_key_pickup(self):
        super().after_key_pickup()
        self.mapping_belief={self.mapping:1.}

    def _marginal(self,weights):
        mass=defaultdict(float)
        for (m,p),w in sorted(weights.items()): mass[m]+=w
        self.mapping_belief=normalize(mass)
        self.posterior_updates+=1

    def act(self,action,expected):
        if not self.monitoring or action>=4:
            return super().act(action,expected)
        origin=self.position
        target=offset(origin,expected)
        tiles=self.map.copy()
        raw=self.mapping.index(action)
        old=self.image
        image=self.raw_step(raw)
        weights=update(hazard_prior(self.mapping_belief or {self.mapping:1.},origin,self.hazard),raw,view(old),view(image),tiles)
        used=set()
        for n in range(self.max_calibration_probes+1):
            if not weights or len(weights)>self.max_calibration_support: break
            poses={p for m,p in weights}
            if len(poses)==1:
                p=next(iter(poses))
                support=sum(tiles.get(offset(p,d))==t for d,t in view(image).items())
                if support>=12 and (len(weights)==1 or (self.mapping,p) in weights):
                    self._marginal(weights)
                    if len(weights)==1 and self.mapping!=next(iter(weights))[0]:
                        self._commit_reidentified(next(iter(weights))[0],p,image)
                    else:
                        self.position=p
                        self._observe(image)
                        if n: self.pose_only_resolutions+=1
                    if p!=target: self.recover_target(target)
                    return
            if n==self.max_calibration_probes: break
            if n==0: self.change_events+=1
            probe=qualified_probe(weights,tiles,used)
            if probe is None: break
            used.add((frozenset(weights),probe))
            previous=image
            self.change_probes+=1
            image=self.raw_step(probe)
            weights=update(weights,probe,view(previous),view(image),tiles)
        raise Unmeasurable('hazard dynamics or pose unresolved')
