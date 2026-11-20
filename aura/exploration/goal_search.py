"""Bounded visual frontier search after passage through a learned doorway."""
import numpy as np
from aura.exploration.door_probe import Unmeasurable, offset
from aura.exploration.movement_calibration import CalibratingCamera
from aura.exploration.visual_frontier import SHIFTS


def stationary_shift(before, after):
    """Require a unique exact stationary-scene alignment; abstain on ties."""
    rr, cc = np.indices((7, 7))
    scores = []
    for dy, dx in SHIFTS:
        r, c = rr + dy, cc + dx
        mask = ((r >= 0) & (r < 7) & (c >= 0) & (c < 7)
                & ~((rr == 3) & (cc == 3)) & ~((r == 3) & (c == 3)))
        scores.append(float(np.mean(np.all(before[:, r[mask], c[mask]] ==
                                          after[:, rr[mask], cc[mask]], axis=0))))
    winners = [s for s, score in zip(SHIFTS, scores) if score == 1.0]
    return winners[0] if len(winners) == 1 else None


class GoalCamera(CalibratingCamera):
    """Experimental exact-alignment fallback, only for anchorless views."""
    def act(self, action, expected):
        from aura.exploration.visual_frontier import visual_displacement
        image = self.step(action)
        before, after = self.anchors(self.image), self.anchors(image)
        votes = {(before[k][0]-after[k][0], before[k][1]-after[k][1])
                 for k in before.keys() & after.keys()}
        if not votes:
            shift = stationary_shift(self.image, image)
            if shift is None:
                shift = visual_displacement(self.image, image)
            votes = {shift}
        if votes != {expected}:
            raise Unmeasurable('movement inconsistent with camera contract')
        self.position = offset(self.position, expected)
        self._observe(image)



def seek_visible_goal(camera, *, search=True):
    """Follow observed paths, exploring new viewpoints if the goal is hidden."""
    visited = {camera.position}
    while True:
        goal = next((point for point,tile in camera.map.items() if tile=='goal'),None)
        if goal is not None:
            # The camera planner normally excludes goal tiles from transit.
            camera.map[goal] = 'empty'
        paths = camera.paths()
        if goal is not None:
            camera.map[goal] = 'goal'
        if goal in paths:
            camera.travel(paths[goal])
            return
        if not search:
            return
        options = [p for p in paths if p not in visited]
        if not options:
            raise Unmeasurable('no unvisited reachable viewpoint')
        def rank(p):
            unknown = sum((p[0]+r,p[1]+c) not in camera.map
                          for r in range(-3,4) for c in range(-3,4))
            return (-unknown/len(paths[p]),len(paths[p]),p)
        target = min(options,key=rank)
        camera.travel(paths[target])
        visited.add(target)


def palette_shift(before, after):
    """Mask off-palette pixels; require a unique zero-conflict alignment."""
    from aura.exploration.door_probe import PALETTE
    rr, cc = np.indices((7,7))
    def known(im):
        return np.array([tuple(p) in PALETTE for p in im.transpose(1,2,0).reshape(-1,3)]).reshape(7,7)
    kb, ka = known(before), known(after)
    candidates = []
    for dy,dx in SHIFTS:
        r,c = rr+dy,cc+dx
        mask = (r>=0)&(r<7)&(c>=0)&(c<7)&~((rr==3)&(cc==3))&~((r==3)&(c==3))
        ar,ac,br,bc = rr[mask],cc[mask],r[mask],c[mask]
        valid = kb[br,bc]&ka[ar,ac]
        n = int(valid.sum())
        m = int(np.count_nonzero(np.any(before[:,br[valid],bc[valid]] != after[:,ar[valid],ac[valid]],axis=0)))
        candidates.append(((dy,dx),n,m))
    winners = [s for s,n,m in candidates if n>=30 and m==0]
    if len(winners)==1 and all(n>=30 and (m>=2 or s==winners[0]) for s,n,m in candidates):
        return winners[0]
    return None


class PaletteGoalCamera(GoalCamera):
    def act(self, action, expected):
        from aura.exploration.visual_frontier import visual_displacement
        image = self.step(action)
        before,after = self.anchors(self.image),self.anchors(image)
        votes = {(before[k][0]-after[k][0],before[k][1]-after[k][1]) for k in before.keys() & after.keys()}
        if not votes:
            shift = stationary_shift(self.image,image)
            if shift is None:
                shift = palette_shift(self.image,image)
            if shift is None:
                shift = visual_displacement(self.image,image)
            votes = {shift}
        if votes != {expected}:
            raise Unmeasurable('movement inconsistent with camera contract')
        self.position = offset(self.position,expected)
        self._observe(image)


class RecoveryGoalCamera(PaletteGoalCamera):
    """Recover obscured calibration axes using previously observed tiles."""
    def after_key_pickup(self):
        from aura.environments.procedural_world import MOVES
        from aura.exploration.door_probe import WALKABLE
        def tile(delta):
            observed = self.at_pixel(delta)
            return observed if observed is not None else self.map.get(offset(self.position, delta))
        action = next((a for a in (0, 2) if all(tile(MOVES[i]) in WALKABLE
                       for i in (a, a ^ 1))), None)
        if action is None:
            raise Unmeasurable('no clear calibration axis')
        image = self.raw_step(action)
        self.probes += 1
        before, after = self.anchors(self.image), self.anchors(image)
        votes = {(before[k][0]-after[k][0], before[k][1]-after[k][1])
                 for k in before.keys() & after.keys()}
        if not votes:
            votes = {palette_shift(self.image, image)}
        if votes == {MOVES[action]}:
            actual = action
        elif votes == {MOVES[action ^ 1]}:
            actual = action ^ 1
        else:
            raise Unmeasurable('ambiguous calibration motion')
        self.reversed = actual != action
        self.position = offset(self.position, MOVES[actual])
        self._observe(image)
        self.travel((actual ^ 1,))


class PassageRecoveryCamera(RecoveryGoalCamera):
    """Verify passage directly if the post-interaction door pixel is missing."""
    def measure(self, arm):
        from aura.environments.procedural_world import MOVES
        enter = self.prepare(arm)
        self.act(4, (0, 0))
        tile = self.at_pixel(MOVES[enter])
        if tile in ('door', 'empty'):
            opened = tile == 'empty'
            self.act(enter, MOVES[enter] if opened else (0, 0))
            return opened
        if tile is not None:
            raise Unmeasurable('unexpected door surface')
        # Missing color is not evidence of an open door. A verified physical
        # displacement into its mapped location is sufficient evidence.
        self.act(enter, MOVES[enter])
        return True


def map_shift(image, position, tiles):
    """Compare candidate poses against previously observed static wall occupancy."""
    from aura.exploration.door_probe import PALETTE
    scores = []
    for shift in SHIFTS:
        pose = offset(position, shift)
        n = conflicts = 0
        for r in range(7):
            for c in range(7):
                if (r,c) == (3,3):
                    continue
                seen = PALETTE.get(tuple(image[:,r,c]))
                prior = tiles.get(offset(pose,(r-3,c-3)))
                if seen is None or prior is None or seen == 'agent':
                    continue
                n += 1
                conflicts += (seen == 'wall') != (prior == 'wall')
        scores.append((shift,n,conflicts))
    winners = [s for s,n,m in scores if n >= 20 and m == 0]
    if len(winners) == 1 and all(n >= 20 and (s == winners[0] or m >= 2) for s,n,m in scores):
        return winners[0]
    return None


class MapRecoveryCamera(PassageRecoveryCamera):
    def act(self, action, expected):
        from aura.exploration.visual_frontier import visual_displacement
        image = self.step(action)
        before, after = self.anchors(self.image), self.anchors(image)
        votes = {(before[k][0]-after[k][0],before[k][1]-after[k][1]) for k in before.keys() & after.keys()}
        if not votes:
            shift = stationary_shift(self.image,image)
            if shift is None:
                shift = palette_shift(self.image,image)
            if shift is None:
                shift = visual_displacement(self.image,image)
            if shift is None:
                shift = map_shift(image,self.position,self.map)
            votes = {shift}
        if votes != {expected}:
            raise Unmeasurable('movement inconsistent with camera contract')
        self.position = offset(self.position,expected)
        self._observe(image)


class ActiveRecoveryCamera(PassageRecoveryCamera):
    """Try another observation using a wall bump shared by both possible poses."""
    def act(self, action, expected):
        from aura.environments.procedural_world import MOVES
        from aura.exploration.visual_frontier import visual_displacement
        prior = self.image
        image = self.step(action)
        def infer(frame):
            before, after = self.anchors(prior), self.anchors(frame)
            votes = {(before[k][0]-after[k][0], before[k][1]-after[k][1])
                     for k in before.keys() & after.keys()}
            if not votes:
                votes = {stationary_shift(prior,frame) or palette_shift(prior,frame)
                         or visual_displacement(prior,frame)}
            return votes
        votes = infer(image)
        if votes == {None} and expected != (0,0):
            poses = (self.position, offset(self.position,expected))
            probe = next((a for a,d in enumerate(MOVES) if all(
                self.map.get(offset(p,d)) == 'wall' for p in poses)),None)
            if probe is not None:
                for _ in range(2):
                    image = self.step(probe)
                    votes = infer(image)
                    if votes != {None}:
                        break
        if votes != {expected}:
            raise Unmeasurable('movement inconsistent with camera contract')
        self.position = offset(self.position,expected)
        self._observe(image)

class BeliefGoalCamera(PassageRecoveryCamera):
    """Experimental posterior fallback after the validated deterministic checks.

    No hidden coordinates are consulted. A posterior that disagrees with the
    expected camera displacement is rejected, never silently integrated.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from aura.exploration.pose_belief import PoseBelief
        self.pose_belief = PoseBelief()
        self.posterior_fallbacks = 0

    def act(self, action, expected):
        from aura.exploration.visual_frontier import visual_displacement
        image = self.step(action)
        before, after = self.anchors(self.image), self.anchors(image)
        votes = {(before[k][0]-after[k][0], before[k][1]-after[k][1])
                 for k in before.keys() & after.keys()}
        if not votes:
            shift = stationary_shift(self.image, image)
            if shift is None:
                shift = palette_shift(self.image, image)
            if shift is None:
                shift = visual_displacement(self.image, image)
            if shift is None:
                estimate = self.pose_belief.update(self.image, image, expected=expected)
                shift = estimate.shift
                if shift is not None:
                    self.posterior_fallbacks += 1
            votes = {shift}
        if votes != {expected}:
            raise Unmeasurable('movement inconsistent with camera contract')
        self.position = offset(self.position, expected)
        self._observe(image)
