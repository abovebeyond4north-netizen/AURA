"""Experimental initial calibration with per-action reset hypotheses."""
from aura.exploration.continuous_hazard import ContinuousHazardCamera, propagate
from aura.exploration.hazard_dynamics import update, qualified_probe
from aura.exploration.learned_dynamics import PERMUTATIONS
from aura.exploration.door_probe import Unmeasurable, offset
from aura.exploration.persistent_pose import view

class ProbabilisticCalibrationCamera(ContinuousHazardCamera):
    def after_key_pickup(self):
        origin=self.position
        tiles=self.map.copy(); tiles[origin]='empty'
        weights={(m,origin):1/24 for m in PERMUTATIONS}
        image=self.image; used=set()
        for n in range(self.max_calibration_probes):
            predicted=propagate(weights,self.hazard)
            if not predicted or len(predicted)>self.max_calibration_support: break
            probe=qualified_probe(predicted,tiles,used)
            if probe is None: break
            used.add((frozenset(predicted),probe))
            previous=image; image=self.raw_step(probe)
            self.probes+=1; self.learning_probes+=1
            weights=update(predicted,probe,view(previous),view(image),tiles)
            self.mapping_support=len({m for m,p in weights})
            if not weights or len(weights)>self.max_calibration_support: break
            poses={p for m,p in weights}
            if len(poses)!=1: continue
            pos=next(iter(poses))
            if sum(tiles.get(offset(pos,d))==t for d,t in view(image).items())<12: continue
            self._marginal(weights)
            best=max(sorted(self.mapping_belief),key=self.mapping_belief.get)
            if self.mapping_belief[best]<.8: continue
            self.mapping=best; self.position=pos; self._observe(image)
            self.map[origin]='empty'; self.monitoring=True
            if pos!=origin:
                key=self.landmarks.pop('key',None)
                try:
                    route=self.paths().get(origin)
                    if route is None: raise Unmeasurable('probabilistic calibration return unavailable')
                    self.travel(route)
                finally:
                    if key is not None: self.landmarks['key']=key
            return
        raise Unmeasurable('probabilistic initial calibration unresolved')
