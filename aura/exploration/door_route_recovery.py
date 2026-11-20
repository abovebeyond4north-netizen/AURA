"""Opt-in recovery into a visually observed open doorway, not a closed one."""
from aura.exploration.continuous_hazard import ContinuousHazardCamera
from aura.exploration.door_probe import Unmeasurable

class DoorRouteRecoveryCamera(ContinuousHazardCamera):
    def __init__(self, image, step):
        self._door_recovery_target = None
        self.door_recoveries = 0
        super().__init__(image, step)

    def paths(self):
        door = self.landmarks.get('door')
        if door is None or door != self._door_recovery_target or self.map.get(door) != 'empty':
            return super().paths()
        del self.landmarks['door']
        try:
            return super().paths()
        finally:
            self.landmarks['door'] = door

    def recover_target(self, target):
        if self._in_travel or target != self.landmarks.get('door') or self.map.get(target) != 'empty':
            return super().recover_target(target)
        self._door_recovery_target = target
        self.door_recoveries += 1
        try:
            route = self.paths().get(target)
            if route is None:
                raise Unmeasurable('observed open doorway unreachable')
            self.travel(route)
            if self.position != target:
                raise Unmeasurable('door recovery destination not reached')
        finally:
            self._door_recovery_target = None


from aura.exploration.probabilistic_calibration import ProbabilisticCalibrationCamera


class CalibratedDoorRecoveryCamera(DoorRouteRecoveryCamera, ProbabilisticCalibrationCamera):
    """Experimental composition; neither parent is promoted."""
