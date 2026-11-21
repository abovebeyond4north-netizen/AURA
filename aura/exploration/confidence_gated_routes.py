"""Conservative observed-landmark gating for an engineered route controller.

Select the early-key route only when both necessary landmarks were directly
seen in the initial RGB frame. Otherwise retain the persistent-pose baseline.
No environment dimensions, noise parameters, or evaluator diagnostics are read.
"""
from aura.exploration.efficient_routes import EfficientRouteCamera
from aura.exploration.persistent_pose import PersistentPoseCamera


class ConfidenceGatedRouteCamera(PersistentPoseCamera):
    def __init__(self, image, step):
        super().__init__(image, step)
        self.early_key_allowed = 'key' in self.landmarks and 'door' in self.landmarks

    def prepare(self, arm):
        if arm == 2 and self.early_key_allowed:
            return EfficientRouteCamera.prepare(self, arm)
        return super().prepare(arm)
