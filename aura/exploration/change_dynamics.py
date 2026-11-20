"""Experimental observation-triggered reidentification of a slowly changing bijection.

No change notification is consumed. After a contradiction, the mapping is assumed
constant for the bounded recovery window; unknown or dynamic cells allow both
blocked and moved outcomes. Known static free cells move deterministically.
"""
from aura.environments.procedural_world import MOVES
from aura.exploration.door_probe import Unmeasurable, WALKABLE, offset
from aura.exploration.learned_dynamics import LearnedDynamicsCamera, initial_support, choose_experiment
from aura.exploration.persistent_pose import view, compatible


def deterministic_support(states, raw, old, new, tiles):
    out = set()
    for mapping, p in states:
        q = offset(p, MOVES[mapping[raw]])
        tile = tiles.get(q)
        options = (p,) if tile == 'wall' else (q,) if tile in WALKABLE | {'goal'} else (p,q)
        out.update((mapping, actual) for actual in options if compatible(old,new,p,actual,tiles))
    return out


class ChangeDynamicsCamera(LearnedDynamicsCamera):
    """Check every post-key movement against all possible displacement hypotheses."""
    def __init__(self, image, step):
        self.monitoring = False
        self.change_events = 0
        self.change_probes = 0
        self.change_resolutions = 0
        self._returning = False
        super().__init__(image, step)

    def after_key_pickup(self):
        super().after_key_pickup()
        self.monitoring = True

    def _commit_reidentified(self, mapping, pos, image):
        self.mapping = mapping
        self.position = pos
        self._observe(image)
        self.change_resolutions += 1

    def accepts_expected(self, poses, states, target, prior, image, expected):
        return poses == {target} and (self.mapping,target) in states

    def recover_target(self, target):
        route=self.paths().get(target)
        if route is None:
            raise Unmeasurable('reidentified pose cannot return to intended route')
        self._returning=True
        try: self.travel(route)
        finally: self._returning=False

    def act(self, action, expected):
        if not self.monitoring or action >= 4:
            return super().act(action, expected)
        origin = self.position
        target = offset(origin, expected)
        tiles = self.map.copy()
        old = self.image
        raw = self.mapping.index(action)
        image = self.raw_step(raw)
        states = deterministic_support(initial_support(origin), raw, view(old), view(image), tiles)
        poses = {pos for _,pos in states}
        if self.accepts_expected(poses, states, target, old, image, expected):
            if sum(tiles.get(offset(target,d)) == t for d,t in view(image).items()) < 12:
                raise Unmeasurable('monitored pose lacks map support')
            self.position = target
            self._observe(image)
            return
        # A contradictory or ambiguous motion is evidence to investigate, not
        # proof of a change. Freeze the shared map until a joint state resolves.
        if self._returning:
            raise Unmeasurable('repeated uncertainty while returning to route')
        self.change_events += 1
        used = set()
        for _ in range(self.max_calibration_probes + 1):
            if not states or len(states)>self.max_calibration_support:
                break
            if len(states)==1:
                mapping,pos=next(iter(states))
                if sum(tiles.get(offset(pos,d))==t for d,t in view(image).items())<12:
                    break
                self._commit_reidentified(mapping,pos,image)
                if pos != target:
                    self.recover_target(target)
                return
            if _ == self.max_calibration_probes:
                break
            probe=choose_experiment(states,tiles,used)
            if probe is None: break
            used.add((frozenset(states),probe))
            previous=image
            self.change_probes+=1
            image=self.raw_step(probe)
            states=deterministic_support(states,probe,view(previous),view(image),tiles)
        raise Unmeasurable('unsignaled dynamics or pose unresolved')


class EvidenceChangeCamera(ChangeDynamicsCamera):
    """Use independently identifiable scene displacement to avoid needless probes.

    No generic expected-action prior or hidden position can override an
    incompatible hypothesis. Palette alignment and stationary landmarks add
    evidence not represented in the static-wall support filter.
    """
    def accepts_expected(self, poses, states, target, prior, image, expected):
        if super().accepts_expected(poses,states,target,prior,image,expected):
            return True
        if (self.mapping,target) not in states:
            return False
        from aura.exploration.goal_search import palette_shift, stationary_shift
        before,after=self.anchors(prior),self.anchors(image)
        votes={(before[k][0]-after[k][0],before[k][1]-after[k][1])
               for k in before.keys() & after.keys() & {'switch','door','goal'}}
        if votes:
            return votes == {expected}
        shift=stationary_shift(prior,image)
        if shift is None: shift=palette_shift(prior,image)
        return shift == expected


class ReplanningChangeCamera(ChangeDynamicsCamera):
    """After verified recovery, replan to the destination, not the aliased edge."""
    max_route_replans = 8

    def __init__(self,image,step):
        self._in_travel=False
        self.route_replans=0
        super().__init__(image,step)

    def recover_target(self,target):
        if not self._in_travel:
            return super().recover_target(target)
        # The outer travel loop can safely replan from a uniquely resolved pose.

    def travel(self,actions):
        if not self.monitoring or self._in_travel:
            return super().travel(actions)
        pending=tuple(actions)
        destination=self.position
        for a in pending:
            destination=offset(destination,MOVES[a])
        replans=0
        self._in_travel=True
        try:
            while pending:
                action,*rest=pending
                target=offset(self.position,MOVES[action])
                self.act(action,MOVES[action])
                if self.position==target:
                    pending=tuple(rest)
                    continue
                if self.position==destination:
                    return
                replans+=1
                self.route_replans+=1
                if replans>self.max_route_replans:
                    raise Unmeasurable('route replanning budget exhausted')
                goal=self.map.get(destination)=='goal'
                if goal: self.map[destination]='empty'
                try: route=self.paths().get(destination)
                finally:
                    if goal: self.map[destination]='goal'
                if route is None:
                    raise Unmeasurable('reidentified pose cannot replan to destination')
                pending=route
        finally:
            self._in_travel=False
