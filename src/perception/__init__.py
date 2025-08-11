"""
Perception module for EV autonomous trucking system.

Implements PointNet++-based feature extractors and perception algorithms.
"""

from .perception_system import PerceptionSystem
from .pointnet import PointNetPlusPlus
from .feature_extractor import FeatureExtractor
from .object_detection import ObjectDetector
from .segmentation import SemanticSegmentation

__all__ = [
    "PerceptionSystem",
    "PointNetPlusPlus", 
    "FeatureExtractor",
    "ObjectDetector",
    "SemanticSegmentation"
]
