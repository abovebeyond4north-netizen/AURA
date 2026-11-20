"""Experimental composition of early-key preparation and persistent localization."""
from aura.exploration.efficient_routes import EfficientRouteCamera
from aura.exploration.persistent_pose import PersistentPoseCamera


class ComposedRouteCamera(EfficientRouteCamera, PersistentPoseCamera):
    """Goal-only preparation with persistent ordinary-motion recovery."""
