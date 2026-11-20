"""Experimental risk-aware frontier ranking, using only the observed map."""
from aura.environments.procedural_world import MOVES
from aura.exploration.door_probe import offset, Unmeasurable
from aura.exploration.observable_routes import ObservableRouteCamera, stationary_contrasts

class AdaptiveRouteCamera(ObservableRouteCamera):
    """Replan at each observation during door survey; rank by route risk."""
    def route_cost(self, actions):
        p = self.position
        total = 0
        for a in actions:
            d = MOVES[a]
            total += 1 + self.alias_penalty * (stationary_contrasts(self.map,p,d)<2)
            p = offset(p,d)
        return total

    def survey(self):
        visits = {self.position: 1}
        while 'door' not in self.landmarks:
            paths = self.paths()
            options = [p for p in paths if p != self.position and visits.get(p,0)==0]
            if not options:
                raise Unmeasurable('closed door outside surveyed area')
            def rank(p):
                gain = sum(offset(p,(r,c)) not in self.map for r in range(-3,4) for c in range(-3,4))
                # Prefer lower risk per new observed cell. Avoid zero-gain targets
                # until all information-bearing routes have been examined.
                cost = self.route_cost(paths[p])
                return (-gain/cost, cost, len(paths[p]), p)
            dest = min(options,key=rank)
            self.travel(paths[dest][:1])
            visits[self.position] = visits.get(self.position,0)+1
        home = self.paths().get((0,0))
        if home is None:
            raise Unmeasurable('cannot return to common start')
        self.travel(home)
