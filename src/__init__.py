"""
EV Autonomous Trucking Perception System

A scalable, cost-effective AI building stack for autonomous driving systems.
"""

__version__ = "1.0.0"
__author__ = "EV Perception Team"
__email__ = "perception@ev.com"

from .perception import PerceptionSystem
from .fusion import MultiModalFusion
from .safety import SafetyFramework
from .ai_tools import AIDevelopmentTools

__all__ = [
    "PerceptionSystem",
    "MultiModalFusion", 
    "SafetyFramework",
    "AIDevelopmentTools"
]
