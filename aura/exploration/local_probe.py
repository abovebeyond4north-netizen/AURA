"""Experimental local-view safe probe fallback; no map writes before resolution."""
from aura.exploration.continuous_hazard import propagate, update, qualified_probe, hazard_prior, view, offset, Unmeasurable
from aura.exploration.frontier_approach import FrontierApproachCamera
from aura.environments.procedural_world import MOVES
from aura.exploration.door_probe import WALKABLE


def local_probe(weights, tiles, used, observed):
    ranked=[]
    for raw in range(4):
        if (frozenset(weights),raw) in used: continue
        outcomes=[]
        for (m,p),w in weights.items():
            d=MOVES[m[raw]]; q=offset(p,d)
            tile=observed.get(d,tiles.get(q))
            if tile not in {'empty','wall'}: break
            outcomes.append((p if tile=='wall' else q,w,tile))
        else:
            if outcomes and any(t=='empty' for _,_,t in outcomes):
                ranked.append((len({q for q,_,_ in outcomes}),sum(w for _,w,t in outcomes if t=='empty'),-raw))
    return -max(ranked)[2] if ranked else None


class LocalProbeCamera(FrontierApproachCamera):
    def act(self, action, expected):
        if not self.monitoring or action >= 4:
            return super().act(action, expected)
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
