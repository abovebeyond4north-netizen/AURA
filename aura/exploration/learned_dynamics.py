"""Experimental online action-effect identification from visual evidence.

Unknown bijection between four raw controls and cardinal displacements. No
simulator positions, labels, or privileged control mapping reach the learner.
The finite permutation prior and static camera model are engineered assumptions.
"""
from collections import Counter
from itertools import permutations
from math import log2
from aura.environments.procedural_world import MOVES
from aura.exploration.door_probe import WALKABLE, Unmeasurable, offset
from aura.exploration.persistent_pose import STATIC, OFFSETS, compatible, view
from aura.exploration.composed_routes import ComposedRouteCamera

PERMUTATIONS = tuple(permutations(range(4)))


def initial_support(origin):
    return {(mapping, origin) for mapping in PERMUTATIONS}


def update_support(states, action, old, new, tiles):
    result = set()
    for mapping, p in states:
        q = offset(p, MOVES[mapping[action]])
        for actual in (p,) if tiles.get(q) == 'wall' else (p, q):
            if compatible(old, new, p, actual, tiles):
                result.add((mapping, actual))
    return result


def choose_experiment(states, tiles, used):
    """Approximate information gain from predicted stationary views.

    Missing tiles never supply contrast. Actions with unknown/dynamic targets
    are excluded. A deterministic tie break makes experiments replayable.
    """
    ranked = []
    for raw in range(4):
        if (frozenset(states), raw) in used:
            continue
        targets = [(p, offset(p, MOVES[m[raw]])) for m,p in sorted(states)]
        if any(tiles.get(q) not in WALKABLE | {'wall'} for p,q in targets):
            continue
        poses = [p if tiles.get(q)=='wall' else q for p,q in targets]
        if all(p == q for (p,_),q in zip(targets,poses)):
            continue
        unique = sorted(set(poses))
        # Count observationally distinguishable pairs, weighted by hypothesis mass.
        counts = Counter(poses)
        separation = 0
        for i,p in enumerate(unique):
            for q in unique[i+1:]:
                contrast = sum(tiles.get(offset(p,d)) in STATIC and
                               tiles.get(offset(q,d)) in STATIC and
                               tiles[offset(p,d)] != tiles[offset(q,d)] for d in OFFSETS)
                separation += counts[p]*counts[q]*min(contrast,3)
        entropy = -sum((n/len(poses))*log2(n/len(poses)) for n in counts.values())
        ranked.append((separation,entropy,-raw))
    return -max(ranked)[2] if ranked else None


class LearnedDynamicsCamera(ComposedRouteCamera):
    max_calibration_probes = 12
    max_calibration_support = 96

    def __init__(self,image,step):
        self.mapping = tuple(range(4))
        self.mapping_support = 24
        self.learning_probes = 0
        super().__init__(image,step)

    def mapped_step(self, action):
        return self.raw_step(self.mapping.index(action) if action < 4 else action)

    def after_key_pickup(self):
        origin = self.position
        tiles = self.map.copy()
        tiles[origin] = 'empty'
        states = initial_support(origin)
        image = self.image
        used = set()
        for _ in range(self.max_calibration_probes):
            action = choose_experiment(states,tiles,used)
            if action is None: break
            used.add((frozenset(states),action))
            previous = image
            image = self.raw_step(action)
            self.probes += 1
            self.learning_probes += 1
            states = update_support(states,action,view(previous),view(image),tiles)
            self.mapping_support = len({m for m,p in states})
            if not states or len(states)>self.max_calibration_support: break
            if len(states)==1:
                mapping,pos = next(iter(states))
                if sum(tiles.get(offset(pos,d))==t for d,t in view(image).items()) < 12:
                    break
                self.mapping = mapping
                self.position = pos
                self._observe(image)
                self.map[origin] = 'empty'
                if pos != origin:
                    key = self.landmarks.pop('key',None)
                    try:
                        route = self.paths().get(origin)
                        if route is None: raise Unmeasurable('learned calibration return unavailable')
                        self.travel(route)
                    finally:
                        if key is not None: self.landmarks['key'] = key
                return
        raise Unmeasurable('learned action mapping unresolved')
