"""Controlled key/switch interventions using the primary world's partial RGB camera.

The public instrument knows the tile palette and cardinal movement contract.
It surveys a closed door, independently varies key collection and switch use,
then tests the door from an adjacent square. No simulator state enters here.
"""
from collections import deque

import numpy as np

from aura.environments.procedural_world import COLORS, MOVES
from aura.exploration.visual_frontier import visual_displacement


class Unmeasurable(Exception):
    """The planned manipulation or outcome could not be checked."""


PALETTE = {value: name for name, value in COLORS.items()}
WALKABLE = {'empty', 'agent', 'switch'}


def offset(point, delta):
    return point[0] + delta[0], point[1] + delta[1]


class Camera:
    def __init__(self, image, step):
        self.step = step
        self.image = image
        self.position = (0, 0)
        self.map = {}
        self.landmarks = {}
        self._observe(image)

    def _observe(self, image):
        if image.shape != (3, 7, 7) or image.dtype != np.uint8:
            raise ValueError('Expected a uint8 egocentric RGB observation')
        self.image = image
        for r, row in enumerate(image.transpose(1, 2, 0).tolist()):
            for c, pixel in enumerate(row):
                tile = PALETTE.get(tuple(pixel))
                if tile is None:
                    continue
                point = offset(self.position, (r - 3, c - 3))
                if tile in ('key', 'switch', 'door'):
                    previous = self.landmarks.setdefault(tile, point)
                    if previous != point:
                        raise Unmeasurable('inconsistent landmark')
                if point != self.position:
                    self.map[point] = tile
        self.map.setdefault(self.position, 'empty')

    def at_pixel(self, delta):
        r, c = offset((3, 3), delta)
        return PALETTE.get(tuple(self.image[:, r, c].tolist()))

    @staticmethod
    def anchors(image):
        # Unique, stationary colored tiles resolve translations on long,
        # otherwise repetitive walls. A disappearing key or door casts no vote.
        anchors = {}
        for tile in ('switch', 'crate', 'door', 'key', 'goal'):
            color = np.asarray(COLORS[tile], dtype=np.uint8)[:, None, None]
            points = np.argwhere(np.all(image == color, axis=0))
            if len(points) == 1:
                anchors[tile] = tuple(map(int, points[0]))
        return anchors

    def act(self, action, expected):
        image = self.step(action)
        before, after = self.anchors(self.image), self.anchors(image)
        votes = {(before[k][0] - after[k][0], before[k][1] - after[k][1])
                 for k in before.keys() & after.keys()}
        if not votes:
            votes = {visual_displacement(self.image, image)}
        if votes != {expected}:
            raise Unmeasurable('movement inconsistent with camera contract')
        self.position = offset(self.position, expected)
        self._observe(image)

    def paths(self):
        """Shortest known routes; never move through a key or an untested door."""
        blocked = {self.landmarks.get(obj) for obj in ('key', 'door')}
        routes = {self.position: ()}
        queue = deque([self.position])
        while queue:
            point = queue.popleft()
            for a, delta in enumerate(MOVES):
                nxt = offset(point, delta)
                if nxt in routes or nxt in blocked or self.map.get(nxt) not in WALKABLE:
                    continue
                routes[nxt] = routes[point] + (a,)
                queue.append(nxt)
        return routes

    def travel(self, actions):
        for a in actions:
            self.act(a, MOVES[a])

    def survey(self):
        # A missing initial sighting is common on larger boards. Explore the
        # reachable part of the map without touching the key or interacting.
        visited = {self.position}
        while 'door' not in self.landmarks:
            paths = self.paths()
            options = [p for p in paths if p not in visited]
            if not options:
                raise Unmeasurable('closed door outside surveyed area')
            def survey_order(p):
                new_cells = sum((p[0]+dy, p[1]+dx) not in self.map
                                for dy in range(-3, 4) for dx in range(-3, 4))
                # Border-following can leave the camera with a uniform wall
                # and no observable horizontal displacement. Prefer interior
                # viewpoints on ties in frontier gain per action.
                border = sum(self.map.get(offset(p, d)) in ('wall', 'unknown') for d in MOVES)
                return (-(new_cells / len(paths[p])), border, len(paths[p]), p)
            dest = min(options, key=survey_order)
            self.travel(paths[dest])
            visited.add(dest)
        home = self.paths().get((0, 0))
        if home is None:
            raise Unmeasurable('cannot return to common start')
        self.travel(home)

    def after_key_pickup(self):
        """Hook for policies that verify their control mapping after pickup."""

    def prepare(self, arm):
        # The preparation contract requires distinct adjacent key/switch tiles
        # and an adjacent wall for a sham action with the same time cost.
        controls = {self.at_pixel(delta): a for a, delta in enumerate(MOVES)}
        for _ in range(2):
            if {'key', 'switch', 'wall'} <= controls.keys() or 'wall' not in controls:
                break
            # Reobserve from the same position with an independently verified
            # wall bump. This also keeps an occluded pixel out of the outcome.
            self.act(controls['wall'], (0, 0))
            for a, delta in enumerate(MOVES):
                tile = self.at_pixel(delta)
                if tile is not None:
                    controls[tile] = a
        if not {'key', 'switch', 'wall'} <= controls.keys():
            raise Unmeasurable('preparation landmarks unavailable')
        self.survey()
        door = self.landmarks['door']
        switch = self.landmarks['switch']
        if sum(map(abs, door)) <= 1:
            raise Unmeasurable('start interaction would also reach the door')

        # Both arms make a verified out-and-back excursion. Crossing the
        # switch without interacting is the sham for crossing the key.
        a = controls['key' if arm & 2 else 'switch']
        self.travel((a,))
        if arm & 2:
            self.after_key_pickup()
        self.travel((a ^ 1,))
        if arm & 1:
            self.act(4, (0, 0))
        else:
            self.act(controls['wall'], (0, 0))

        paths = self.paths()
        vantage = []
        for a, delta in enumerate(MOVES):
            point = (door[0] - delta[0], door[1] - delta[1])
            distance_to_switch = abs(point[0] - switch[0]) + abs(point[1] - switch[1])
            if point in paths and distance_to_switch > 1:
                vantage.append((len(paths[point]), a, paths[point]))
        if not vantage:
            raise Unmeasurable('no isolated approach to the door')
        _, enter, route = min(vantage)
        self.travel(route)
        return enter

    def measure(self, arm):
        enter = self.prepare(arm)
        self.act(4, (0, 0))  # identical final door interaction in every arm
        tile = self.at_pixel(MOVES[enter])
        if tile not in ('door', 'empty'):
            raise Unmeasurable('door surface occluded')
        # Require agreement between the material seen and physical passage.
        opened = tile == 'empty'
        self.act(enter, MOVES[enter] if opened else (0, 0))
        return opened


def measure_door(initial, step, arm):
    if type(arm) is not int or not 0 <= arm < 4:
        raise ValueError('Invalid factorial arm')
    try:
        return Camera(initial, step).measure(arm), None
    except Unmeasurable as exc:
        return None, str(exc)
