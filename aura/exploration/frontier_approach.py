"""Bounded visual search for a missing known-safe door approach (goal only)."""
from aura.environments.procedural_world import MOVES
from aura.exploration.door_probe import Unmeasurable, offset
from aura.exploration.door_route_recovery import CalibratedDoorRecoveryCamera


def isolated_approaches(door, switch, paths, route_cost):
    approaches = []
    for action, delta in enumerate(MOVES):
        point = offset(door, (-delta[0], -delta[1]))
        if point in paths and (switch is None or sum(abs(a-b) for a,b in zip(point,switch)) > 1):
            approaches.append((route_cost(paths[point]), action, paths[point]))
    return approaches


class FrontierApproachCamera(CalibratedDoorRecoveryCamera):
    """Survey reachable viewpoints after first door sighting; goal-only protocol."""
    max_approach_views = 8

    def __init__(self, image, step):
        self.approach_views = 0
        super().__init__(image, step)

    def prepare(self, arm):
        try:
            return super().prepare(arm)
        except Unmeasurable as exc:
            if str(exc) != 'no isolated approach to the door':
                raise
        return self.discover_approach()

    def discover_approach(self):
        door = self.landmarks['door']
        visited = {self.position}
        for _ in range(self.max_approach_views + 1):
            paths = self.paths()
            approaches = isolated_approaches(door, self.landmarks.get('switch'), paths, self.route_cost)
            if approaches:
                _, action, route = min(approaches)
                self.travel(route)
                return action
            if self.approach_views >= self.max_approach_views:
                break
            options = [p for p in paths if p not in visited and paths[p]]
            if not options:
                break
            def rank(p):
                cost = self.route_cost(paths[p])
                gain = sum(offset(p,(r,c)) not in self.map for r in range(-3,4) for c in range(-3,4))
                distance = sum(abs(a-b) for a,b in zip(p,door))
                return (-gain/cost, distance, cost, p)
            point = min(options, key=rank)
            self.travel(paths[point])
            visited.add(self.position)
            self.approach_views += 1
        raise Unmeasurable('bounded approach frontier exhausted')
