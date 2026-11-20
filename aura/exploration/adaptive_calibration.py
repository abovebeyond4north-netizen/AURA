"""Experimental joint control-map/pose calibration with bounded raw probes."""
from aura.environments.procedural_world import MOVES
from aura.exploration.door_probe import Unmeasurable, WALKABLE, offset
from aura.exploration.goal_search import palette_shift
from aura.exploration.persistent_pose import compatible, view, STATIC, OFFSETS, PersistentPoseCamera
from aura.exploration.composed_routes import ComposedRouteCamera


def advance(states, action, before, after, tiles):
    result = set()
    for reverse, p in states:
        q = offset(p, MOVES[action ^ int(reverse)])
        options = (p,) if tiles.get(q) == 'wall' else (p, q)
        result.update((reverse, x) for x in options if compatible(before, after, p, x, tiles))
    return result


def choose_raw(states, tiles, used):
    best = None
    for a in range(4):
        if (frozenset(states), a) in used:
            continue
        targets = [(p, offset(p, MOVES[a ^ int(rev)])) for rev, p in sorted(states)]
        if any(tiles.get(q) not in WALKABLE | {'wall'} for p, q in targets):
            continue
        predicted = [p if tiles.get(q) == 'wall' else q for p, q in targets]
        if all(p == q for (p, _), q in zip(targets, predicted)):
            continue
        scores = [sum(tiles.get(offset(p,d)) in STATIC and tiles.get(offset(q,d)) in STATIC and tiles[offset(p,d)] != tiles[offset(q,d)] for d in OFFSETS) for i,p in enumerate(predicted) for q in predicted[i+1:]]
        rank = (sum(v > 0 for v in scores), sum(scores), -a)
        if best is None or rank > best[0]:
            best = rank, a
    return None if best is None else best[1]


class AdaptiveCalibration:
    max_calibration_probes = 4

    def after_key_pickup(self):
        origin = self.position
        def tile(d):
            observed = self.at_pixel(d)
            return observed if observed is not None else self.map.get(offset(origin,d))
        action = next((a for a in (0,2) if all(tile(MOVES[i]) in WALKABLE for i in (a,a^1))), None)
        states = {(False, origin), (True, origin)}
        tiles = self.map.copy()
        tiles[origin] = 'empty'
        used = set()
        image = self.image
        for i in range(self.max_calibration_probes):
            if action is None:
                action = choose_raw(states, tiles, used)
            if action is None:
                break
            used.add((frozenset(states), action))
            prior = image
            image = self.raw_step(action)
            self.probes += 1
            before, after = self.anchors(prior), self.anchors(image)
            votes = {(before[k][0]-after[k][0], before[k][1]-after[k][1]) for k in before.keys() & after.keys()}
            if not votes:
                votes = {palette_shift(prior,image)}
            if i == 0 and all(tile(MOVES[j]) in WALKABLE for j in (action,action^1)):
                if votes in ({MOVES[action]}, {MOVES[action^1]}):
                    actual = action if votes == {MOVES[action]} else action^1
                    self.reversed = actual != action
                    self.position = offset(origin,MOVES[actual])
                    self._observe(image)
                    self.travel((actual^1,))
                    return
            states = advance(states,action,view(prior),view(image),tiles)
            if len(states)==1:
                reverse, pos = next(iter(states))
                if sum(tiles.get(offset(pos,d))==t for d,t in view(image).items()) < 12:
                    break
                self.reversed = reverse
                self.position = pos
                self._observe(image)
                self.map[origin] = 'empty'
                if pos != origin:
                    key = self.landmarks.pop('key',None)
                    try:
                        route = self.paths().get(origin)
                        if route is None: raise Unmeasurable('calibration return unavailable')
                        self.travel(route)
                    finally:
                        if key is not None: self.landmarks['key'] = key
                return
            if not states or len(states)>12:
                break
            action = None
        raise Unmeasurable('joint calibration unresolved')


class CalibratedComposedCamera(AdaptiveCalibration, ComposedRouteCamera):
    pass


class CalibratedPersistentCamera(AdaptiveCalibration, PersistentPoseCamera):
    pass
