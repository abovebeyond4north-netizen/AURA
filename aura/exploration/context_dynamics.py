"""Experimental context-conditioned movement identification after verified passage.

The context boundary is engineered; whether the mapping changes and its value
are not provided. Recalibration never commits an unresolved pose or mapping.
"""
from aura.exploration.door_probe import Unmeasurable, offset
from aura.exploration.learned_dynamics import (
    LearnedDynamicsCamera, initial_support, choose_experiment, update_support,
)
from aura.exploration.persistent_pose import view


class ContextDynamicsCamera(LearnedDynamicsCamera):
    def __init__(self, image, step):
        self.context_probes = 0
        self.context_calibrations = 0
        super().__init__(image, step)

    def after_verified_passage(self):
        origin = self.position
        tiles = self.map.copy()
        tiles[origin] = 'empty'
        states = initial_support(origin)
        image = self.image
        used = set()
        for _ in range(self.max_calibration_probes):
            action = choose_experiment(states, tiles, used)
            if action is None:
                break
            used.add((frozenset(states), action))
            previous = image
            self.context_probes += 1
            image = self.raw_step(action)
            states = update_support(states, action, view(previous), view(image), tiles)
            if not states or len(states) > self.max_calibration_support:
                break
            if len(states) == 1:
                mapping, pos = next(iter(states))
                if sum(tiles.get(offset(pos,d)) == t for d,t in view(image).items()) < 12:
                    break
                self.mapping = mapping
                self.position = pos
                self.map[origin] = 'empty'
                self._observe(image)
                self.context_calibrations += 1
                return
        raise Unmeasurable('post-passage action mapping unresolved')
