"""Experimental per-probe reset propagation; no support pruning."""
from collections import defaultdict
from aura.exploration.hazard_dynamics import HazardDynamicsCamera, normalize, update, qualified_probe, hazard_prior
from aura.exploration.learned_dynamics import PERMUTATIONS
from aura.exploration.door_probe import Unmeasurable, offset
from aura.exploration.persistent_pose import view


def propagate(weights, hazard):
    if not 0 < hazard <= 1:
        raise ValueError('invalid hazard')
    poses = defaultdict(float)
    for (m,p),w in weights.items():
        poses[p] += w
    return normalize({(m,p):(1-hazard)*weights.get((m,p),0)+hazard*w/len(PERMUTATIONS)
                      for p,w in poses.items() for m in PERMUTATIONS})


class ContinuousHazardCamera(HazardDynamicsCamera):
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
            if probe is None: break
            used.add((frozenset(predicted),probe))
            previous=image
            self.change_probes+=1
            image=self.raw_step(probe)
            weights=update(predicted,probe,view(previous),view(image),tiles)
        raise Unmeasurable('continuous hazard pose unresolved')
