"""
EV Autonomous Trucking Perception System

A scalable, cost-effective AI building stack for autonomous driving systems.
"""

__version__ = "1.0.0"
__author__ = "EV Perception Team"
__email__ = "perception@ev.com"

__all__ = [
    "PerceptionSystem",
    "MultiModalFusion",
    "SafetyFramework",
    "AIDevelopmentTools",
]


def __getattr__(name):
    """Load top-level systems on demand so a submodule import stays small."""
    if name == "PerceptionSystem":
        from .perception import PerceptionSystem
        return PerceptionSystem
    if name == "MultiModalFusion":
        from .fusion import MultiModalFusion
        return MultiModalFusion
    if name == "SafetyFramework":
        from .safety import SafetyFramework
        return SafetyFramework
    if name == "AIDevelopmentTools":
        from .ai_tools import AIDevelopmentTools
        return AIDevelopmentTools
    raise AttributeError(name)
