"""One RGB displacement probe to select between normal and reversed controls."""
from aura.environments.procedural_world import MOVES
from aura.exploration.door_probe import Camera, Unmeasurable, WALKABLE, offset


class CalibratingCamera(Camera):
    def __init__(self, image, step):
        self.raw_step = step
        self.reversed = False
        self.probes = 0
        super().__init__(image, self.mapped_step)

    def mapped_step(self, action):
        return self.raw_step(action ^ 1 if action < 4 and self.reversed else action)

    def after_key_pickup(self):
        # Select an axis with clear space in both directions. No effect can
        # be attributed to reversal if one of those movements is blocked.
        action = next((a for a in (0,2)
                       if all(self.at_pixel(MOVES[i]) in WALKABLE for i in (a,a^1))),None)
        if action is None:
            raise Unmeasurable('no clear calibration axis')
        image = self.raw_step(action)
        self.probes += 1
        before, after = self.anchors(self.image), self.anchors(image)
        shifts = {(before[k][0]-after[k][0],before[k][1]-after[k][1])
                  for k in before.keys() & after.keys()}
        if shifts == {MOVES[action]}:
            actual = action
        elif shifts == {MOVES[action^1]}:
            actual = action^1
        else:
            raise Unmeasurable('ambiguous calibration motion')
        self.reversed = actual != action
        self.position = offset(self.position,MOVES[actual])
        self._observe(image)
        self.travel((actual^1,))
