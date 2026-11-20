"""Experimental bounded set-valued localization with safe active probes.

The set is persistent across actions, not a calibrated Bayesian posterior.
No candidate-specific writes enter the shared map before resolution.
"""
from itertools import combinations
import numpy as np
from aura.environments.procedural_world import MOVES
from aura.exploration.door_probe import PALETTE, WALKABLE, Unmeasurable, offset
from aura.exploration.goal_search import stationary_shift, palette_shift
from aura.exploration.visual_frontier import visual_displacement
from aura.exploration.adaptive_routes import AdaptiveRouteCamera

STATIC = frozenset(('wall','unknown','empty','switch','goal'))
OFFSETS = tuple((r,c) for r in range(-3,4) for c in range(-3,4) if (r,c)!=(0,0))


def view(frame):
    return {d: tile for d in OFFSETS if (tile := PALETTE.get(tuple(frame[:,d[0]+3,d[1]+3]))) in STATIC}


def compatible(old, new, p, q, tiles):
    """Compare invariant observed tiles; erased and dynamic pixels are not evidence."""
    delta = (q[0]-p[0],q[1]-p[1])
    for d,t in new.items():
        a = old.get(offset(d,delta))
        b = tiles.get(offset(q,d))
        for prior in (a,b):
            if prior in STATIC:
                if prior != t:
                    return False
    # Missing evidence cannot eliminate a physically possible pose.
    return True


def propagate(poses, direction, old, new, tiles):
    """Keep both moved and blocked outcomes for known-free cells."""
    result=set()
    for p in poses:
        q=offset(p,direction)
        options=(p,) if tiles.get(q)=='wall' else (p,q)
        result.update(q for q in options if compatible(old,new,p,q,tiles))
    return result


def select_probe(poses, tiles, used):
    """Rank safe probes by predicted map contrast, never by true pose."""
    best=None
    for a,d in enumerate(MOVES):
        if (frozenset(poses),a) in used:
            continue
        targets=[offset(p,d) for p in sorted(poses)]
        if any(tiles.get(q) not in WALKABLE | {'wall'} for q in targets):
            continue
        predictions=[p if tiles.get(q)=='wall' else q for p,q in zip(sorted(poses),targets)]
        if predictions == sorted(poses):
            continue
        contrasts=[]
        for p,q in combinations(predictions,2):
            contrasts.append(sum(tiles.get(offset(p,x)) in STATIC and tiles.get(offset(q,x)) in STATIC and tiles[offset(p,x)]!=tiles[offset(q,x)] for x in OFFSETS))
        score=(sum(c>0 for c in contrasts),sum(contrasts),-a)
        if best is None or score>best[0]: best=score,a
    return None if best is None else best[1]


class PersistentPoseCamera(AdaptiveRouteCamera):
    max_probes=4
    max_poses=12
    def __init__(self,image,step):
        super().__init__(image,step)
        self.recoveries=0
        self.resolutions=0
        self.probe_actions=0
        self.max_support=0
        self._recovering=False

    def act(self,action,expected):
        prior=self.image
        image=self.step(action)
        before,after=self.anchors(prior),self.anchors(image)
        votes={(before[k][0]-after[k][0],before[k][1]-after[k][1]) for k in before.keys() & after.keys()}
        if not votes:
            shift=stationary_shift(prior,image)
            if shift is None: shift=palette_shift(prior,image)
            if shift is None: shift=visual_displacement(prior,image)
            votes={shift}
        if votes=={expected}:
            self.position=offset(self.position,expected)
            self._observe(image)
            return
        target=offset(self.position,expected)
        # Probe only ordinary mapped movements. Key pickup and door traversal
        # may change control or world state; never speculate about those.
        if (self._recovering or action>=4 or expected!=MOVES[action] or
                self.map.get(target) not in WALKABLE or
                any(v not in (None,(0,0),expected) for v in votes)):
            raise Unmeasurable('movement inconsistent with camera contract')
        self.recoveries+=1
        poses=propagate({self.position},expected,view(prior),view(image),self.map)
        used=set()
        for _ in range(self.max_probes):
            self.max_support=max(self.max_support,len(poses))
            if len(poses)<=1 or len(poses)>self.max_poses: break
            probe=select_probe(poses,self.map,used)
            if probe is None: break
            used.add((frozenset(poses),probe))
            previous=image
            self.probe_actions+=1
            image=self.step(probe)
            poses=propagate(poses,MOVES[probe],view(previous),view(image),self.map)
        self.max_support=max(self.max_support,len(poses))
        if len(poses)!=1:
            raise Unmeasurable('persistent pose unresolved')
        resolved=next(iter(poses))
        if sum(self.map.get(offset(resolved,d))==t for d,t in view(image).items())<12:
            raise Unmeasurable('persistent pose lacks map support')
        self.position=resolved
        self._observe(image)
        self.resolutions+=1
        if self.position!=target:
            route=self.paths().get(target)
            if route is None: raise Unmeasurable('resolved pose cannot return to intended route')
            self._recovering=True
            try: self.travel(route)
            finally: self._recovering=False
