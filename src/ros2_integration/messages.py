"""
Sensor contracts shared by physical, replay, and simulated sources.

These are the in-process equivalents of ROS 2 topics and QoS. A real rclpy
graph can carry the same fields; the software-in-the-loop loop does not import
ROS 2.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class QoSProfile:
    """Starting QoS choice for one stream. Production still has to measure it."""

    reliability: str
    history: str
    depth: int
    durability: str
    deadline_ms: Optional[float] = None


# High-rate images: a stale frame is usually less useful than the newest one.
CAMERA_QOS = QoSProfile("BEST_EFFORT", "KEEP_LAST", 1, "VOLATILE", deadline_ms=50)
# Scans are loss-sensitive for tracking, but the queue stays bounded.
LIDAR_QOS = QoSProfile("RELIABLE", "KEEP_LAST", 5, "VOLATILE", deadline_ms=100)
RADAR_QOS = QoSProfile("BEST_EFFORT", "KEEP_LAST", 5, "VOLATILE", deadline_ms=50)
IMU_QOS = QoSProfile("BEST_EFFORT", "KEEP_LAST", 20, "VOLATILE", deadline_ms=10)
# Control commands are not a sensor stream. Dropping one can be safety-relevant.
CONTROL_QOS = QoSProfile("RELIABLE", "KEEP_LAST", 10, "VOLATILE", deadline_ms=20)
# Late subscribers still need the current static map or calibration.
MAP_QOS = QoSProfile("RELIABLE", "KEEP_LAST", 1, "TRANSIENT_LOCAL")

TOPIC_QOS = {
    "/camera/front/image": CAMERA_QOS,
    "/lidar/points": LIDAR_QOS,
    "/radar/tracks": RADAR_QOS,
    "/imu/data": IMU_QOS,
    "/vehicle/command": CONTROL_QOS,
    "/map/static": MAP_QOS,
}


@dataclass
class SensorSample:
    """One acquisition, stamped when the sensor measured it, not when it arrived."""

    topic: str
    modality: str
    stamp_ns: int
    frame_id: str
    sequence: int
    source: str
    valid: bool
    payload: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
