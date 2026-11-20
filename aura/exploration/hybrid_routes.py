"""Experimental pre-action strategy selection from observable stationary structure."""
from aura.exploration.door_probe import PALETTE
from aura.exploration.adaptive_routes import AdaptiveRouteCamera
from aura.exploration.efficient_routes import EfficientRouteCamera


def choose_route(image):
    """Early pickup only in high-support views (at least 11 visible wall tiles).

    Threshold chosen on public development geometries; not a learned rule.
    Unknown colors cannot contribute to stationary structure support.
    """
    tiles = [PALETTE.get(tuple(pixel)) for pixel in image.transpose(1,2,0).reshape(-1,3)]
    return 'early' if tiles.count('wall') >= 11 else 'survey'


class HybridRouteCamera(AdaptiveRouteCamera):
    def __init__(self,image,step):
        super().__init__(image,step)
        self.strategy = choose_route(image)

    def prepare(self, arm):
        if arm != 2:
            raise ValueError('goal-only controller supports key arm 2 only')
        if self.strategy == 'early':
            return EfficientRouteCamera.prepare(self,arm)
        return super().prepare(arm)
