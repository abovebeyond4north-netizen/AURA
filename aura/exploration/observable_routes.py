"""Experimental route selection using only observed tile geometry.

A low-evidence edge is costly, not forbidden. Motion verification is unchanged.
"""
from heapq import heappop, heappush
from aura.environments.procedural_world import MOVES
from aura.exploration.door_probe import offset, WALKABLE
from aura.exploration.goal_search import PassageRecoveryCamera

STATIC = {'wall', 'unknown', 'empty', 'switch', 'goal'}


def stationary_contrasts(tiles, origin, delta):
    """Count known distinct stationary tiles at equal camera pixels.

    Excludes center and the previous marker's projected location, and never
    interprets an unobserved tile as free space or a known boundary.
    """
    dest = offset(origin, delta)
    contrasts = 0
    for r in range(-3, 4):
        for c in range(-3, 4):
            if (r, c) in ((0, 0), (-delta[0], -delta[1])):
                continue
            a = tiles.get(offset(origin, (r, c)))
            b = tiles.get(offset(dest, (r, c)))
            if a in STATIC and b in STATIC and a != b:
                contrasts += 1
    return contrasts


class ObservableRouteCamera(PassageRecoveryCamera):
    """Penalize visually aliased edges; preserve all v50 motion checks."""
    alias_penalty = 8

    def paths(self):
        blocked = {self.landmarks.get(obj) for obj in ('key', 'door')}
        routes = {self.position: ()}
        costs = {self.position: 0}
        queue = [(0, (), self.position)]
        while queue:
            cost, path, p = heappop(queue)
            if (cost, path) != (costs[p], routes[p]):
                continue
            for a, d in enumerate(MOVES):
                nxt = offset(p, d)
                if nxt in blocked or self.map.get(nxt) not in WALKABLE:
                    continue
                penalty = self.alias_penalty if stationary_contrasts(self.map, p, d) < 2 else 0
                new_cost, new_path = cost + 1 + penalty, path + (a,)
                if nxt not in costs or (new_cost,new_path) < (costs[nxt],routes[nxt]):
                    costs[nxt], routes[nxt] = new_cost,new_path
                    heappush(queue,(new_cost,new_path,nxt))
        return routes
