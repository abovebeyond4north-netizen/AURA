"""Experimental two-step conservative map-contrast lookahead."""
from collections import defaultdict
from aura.exploration.local_probe import LocalProbeCamera, local_probe
from aura.exploration.continuous_hazard import propagate
from aura.exploration.hazard_dynamics import normalize
from aura.exploration.persistent_pose import OFFSETS, STATIC
from aura.environments.procedural_world import MOVES
from aura.exploration.door_probe import offset


def transition(weights, tiles, raw, observed=None):
    out=defaultdict(float)
    moving=False
    for (m,p),w in sorted(weights.items()):
        d=MOVES[m[raw]]; q=offset(p,d)
        tile=(observed or {}).get(d,tiles.get(q))
        if tile not in {'empty','wall'}: return None
        if tile=='wall': q=p
        moving |= q!=p
        out[m,q]+=w
    return normalize(out) if moving else None


def contrast(weights, tiles):
    mass=defaultdict(float)
    for (_,p),w in weights.items(): mass[p]+=w
    score=0.
    for d in OFFSETS:
        counts=defaultdict(float)
        for p,w in mass.items():
            t=tiles.get(offset(p,d))
            if t in STATIC: counts[t]+=w
        score+=sum(a*b for i,a in enumerate(counts.values()) for b in list(counts.values())[i+1:])
    return score


def lookahead(weights,tiles,used,observed,hazard):
    ranked=[]
    for raw in range(4):
        if (frozenset(weights),raw) in used: continue
        nxt=transition(weights,tiles,raw,observed)
        if not nxt: continue
        future=propagate(nxt,hazard)
        scores=[contrast(y,tiles) for a in range(4) if (y:=transition(future,tiles,a))]
        ranked.append((contrast(nxt,tiles)+.5*max(scores,default=0),-raw))
    return -max(ranked)[1] if ranked and max(ranked)[0]>0 else None

from aura.exploration.local_probe import FrontierApproachCamera, update, qualified_probe, hazard_prior, view, Unmeasurable

class BeliefLookaheadCamera(LocalProbeCamera):
    def act(self, action, expected):
        if not self.monitoring or action >= 4:
            return FrontierApproachCamera.act(self, action, expected)
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
                if support>=12:
                    self._marginal(weights)
                    # Operational action selection is a hypothesis, not a certified mapping.
                    self.mapping=max(sorted(self.mapping_belief),key=self.mapping_belief.get)
                    self.position=p
                    self._observe(image)
                    if n: self.pose_only_resolutions+=1
                    if p!=target: self.recover_target(target)
                    return
            if n==self.max_calibration_probes: break
            if n==0: self.change_events+=1
            predicted=propagate(weights,self.hazard)
            if len(predicted)>self.max_calibration_support: break
            probe=lookahead(predicted,tiles,used,view(image),self.hazard)
            if probe is None:
                probe=qualified_probe(predicted,tiles,used)
            if probe is None:
                probe=local_probe(predicted,tiles,used,view(image))
            if probe is None: break
            used.add((frozenset(predicted),probe))
            previous=image
            self.change_probes+=1
            image=self.raw_step(probe)
            weights=update(predicted,probe,view(previous),view(image),tiles)
        raise Unmeasurable('continuous hazard pose unresolved')
