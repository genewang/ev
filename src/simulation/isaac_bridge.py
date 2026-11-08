"""
Isaac Sim software-in-the-loop stand-in.

NVIDIA Isaac Sim is not bundled and is not launched here. This module emits the
same topic names, acquisition stamps, calibration, and label schema that the
ROS 2 bridge would publish, and it records the scenario provenance a later
Isaac run needs: simulator version, scene, seed, weather, and label schema.

Swap ``backend`` to ``isaac-ros2-bridge`` only when a real Omniverse session is
publishing these topics. Perception code should depend on ``SensorSample``,
not on which backend filled it.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List

from ..ros2_integration.graph import InProcessGraph
from ..ros2_integration.messages import SensorSample


TOPIC_BY_MODALITY = {
    "camera": "/camera/front/image",
    "lidar": "/lidar/points",
    "radar": "/radar/tracks",
    "imu": "/imu/data",
}

# Acquisition offsets from the frame clock. Values are inside a 20 ms slop.
ACQUISITION_OFFSET_NS = {
    "camera": 0,
    "lidar": 8_000_000,
    "radar": 4_000_000,
    "imu": 1_000_000,
}


@dataclass
class ScenarioProvenance:
    simulator: str = "isaac-stand-in"
    simulator_version: str = "0"
    scene: str = "straight_road"
    seed: int = 0
    weather: str = "clear"
    lighting: str = "day"
    calibration_id: str = "truck_front_v1"
    label_schema: str = "box3d-v1"
    backend: str = "stand-in"


@dataclass
class SimActor:
    actor_id: str
    cls: str
    position: tuple
    velocity: tuple


@dataclass
class SimFrame:
    frame_index: int
    stamp_ns: int
    actors: List[SimActor]
    provenance: ScenarioProvenance
    samples: List[SensorSample] = field(default_factory=list)


class IsaacRosBridge:
    """Publish one simulated frame onto the shared in-process graph."""

    def __init__(self, graph: InProcessGraph, provenance: ScenarioProvenance = None):
        self.graph = graph
        self.provenance = provenance or ScenarioProvenance()

    def publish_frame(self, frame: SimFrame, stale_camera: bool = False) -> None:
        meta = asdict(frame.provenance)
        for modality, topic in TOPIC_BY_MODALITY.items():
            offset = ACQUISITION_OFFSET_NS[modality]
            if stale_camera and modality == "camera":
                offset = 40_000_000
            sample = SensorSample(
                topic=topic,
                modality=modality,
                stamp_ns=frame.stamp_ns + offset,
                frame_id="base_link",
                sequence=frame.frame_index,
                source=frame.provenance.simulator,
                valid=True,
                payload=self._payload(modality, frame),
                provenance=meta,
            )
            frame.samples.append(sample)
            self.graph.publish(sample)

    def _payload(self, modality: str, frame: SimFrame) -> Dict:
        if modality == "lidar":
            return {"num_points": 1024, "actors": [actor.actor_id for actor in frame.actors]}
        if modality == "camera":
            return {"encoding": "rgb8", "width": 640, "height": 480}
        if modality == "radar":
            return {
                "tracks": [
                    {"id": actor.actor_id, "position": actor.position, "velocity": actor.velocity}
                    for actor in frame.actors
                ]
            }
        return {"angular_velocity": (0.0, 0.0, 0.0), "linear_acceleration": (0.0, 0.0, 9.81)}


def make_frame(index: int, provenance: ScenarioProvenance, dt_ns: int = 33_000_000) -> SimFrame:
    """One pedestrian crossing at constant lateral speed. Ground truth is known."""
    stamp = index * dt_ns
    x = -5.0 + index * 0.4
    actor = SimActor("ped-1", "pedestrian", (x, 1.5, 0.0), (0.4 * 1e9 / dt_ns, 0.0, 0.0))
    return SimFrame(index, stamp, [actor], provenance)
