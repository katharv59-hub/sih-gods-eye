"""Camera graph — CameraNode topology, transition modeling.

Phase 3 deliverable: Multi-camera awareness and transition priors.
"""

from gods_eye.camera_graph.camera_graph import CameraGraph
from gods_eye.camera_graph.config_loader import load_camera_config

__all__ = ["CameraGraph", "load_camera_config"]
