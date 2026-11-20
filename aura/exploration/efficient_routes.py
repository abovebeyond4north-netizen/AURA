"""Goal-only route controller; does not replace matched causal interventions."""
from aura.environments.procedural_world import MOVES
from aura.exploration.door_probe import Unmeasurable, offset
from aura.exploration.adaptive_routes import AdaptiveRouteCamera


class EfficientRouteCamera(AdaptiveRouteCamera):
    """Collect the observed adjacent key before surveying, avoiding a round trip.

    The key-door protocol is supplied by the caller (arm 2), not discovered.
    This controller is not a valid factorial causal measurement instrument.
    """
    def prepare(self, arm):
        if arm != 2:
            raise ValueError('goal-only controller supports key arm 2 only')
        controls = {self.at_pixel(delta): a for a, delta in enumerate(MOVES)}
        for _ in range(2):
            if 'key' in controls or 'wall' not in controls:
                break
            self.act(controls['wall'], (0, 0))
            controls.update({self.at_pixel(d): a for a,d in enumerate(MOVES) if self.at_pixel(d) is not None})
        if 'key' not in controls:
            raise Unmeasurable('preparation key unavailable')
        self.travel((controls['key'],))
        self.after_key_pickup()
        # The collected key is no longer a blocking landmark. Its cached tile
        # is replaced only when the return from calibration was verified.
        self.landmarks.pop('key',None)
        self.map[self.position] = 'empty'
        visited = {self.position}
        while 'door' not in self.landmarks:
            paths = self.paths()
            options = [p for p in paths if p not in visited]
            if not options:
                raise Unmeasurable('closed door outside surveyed area')
            def rank(p):
                gain = sum(offset(p,(r,c)) not in self.map for r in range(-3,4) for c in range(-3,4))
                cost = self.route_cost(paths[p])
                return (-gain/cost,cost,len(paths[p]),p)
            dest = min(options,key=rank)
            self.travel(paths[dest][:1])
            visited.add(self.position)
        door = self.landmarks['door']
        switch = self.landmarks.get('switch')
        paths = self.paths()
        approaches = []
        for a,d in enumerate(MOVES):
            point = offset(door,(-d[0],-d[1]))
            if point in paths and (switch is None or sum(abs(x-y) for x,y in zip(point,switch))>1):
                approaches.append((self.route_cost(paths[point]),a,paths[point]))
        if not approaches:
            raise Unmeasurable('no isolated approach to the door')
        _,enter,route = min(approaches)
        self.travel(route)
        return enter
