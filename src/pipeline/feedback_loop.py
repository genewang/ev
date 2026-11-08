"""
Software-in-the-loop feedback loop.

Physical or simulated sensors publish one message contract. A synchronizer
aligns them by acquisition time. Context-conditioned attention fuses the
modalities that arrived inside the slop. A constant-velocity tracker associates
ground-truth actors across frames. Stage times are measured and labeled as this
in-process loop, not as a certified vehicle latency.

Isaac Sim itself is not started. ``IsaacRosBridge`` is the seam where a real
ROS 2 bridge would publish the same topics.
"""

import time
from typing import Dict, List

import torch

from ..fusion.context_attention import ContextConditionedAttention
from ..fusion.tracking import ConstantVelocityTracker, Detection
from ..inference.benchmark import benchmark_callable
from ..inference.targets import DeploymentTarget, allows_perception
from ..simulation.isaac_bridge import IsaacRosBridge, ScenarioProvenance, make_frame
from ..ros2_integration.graph import InProcessGraph
from ..ros2_integration.sync import TimestampSynchronizer


MODALITIES = ["camera", "lidar", "radar", "imu"]
FEATURE_DIMS = {"lidar": 8, "camera": 8, "radar": 8, "imu": 8}


def run_feedback_loop(frames: int = 6, seed: int = 7, slop_ns: int = 20_000_000) -> Dict:
    provenance = ScenarioProvenance(seed=seed, weather="rain", lighting="dusk")
    graph = InProcessGraph()
    bridge = IsaacRosBridge(graph, provenance)
    sync = TimestampSynchronizer(MODALITIES, slop_ns=slop_ns, anchor="lidar")
    model = ContextConditionedAttention(FEATURE_DIMS, embed_dim=32, num_heads=4)
    model.eval()
    tracker = ConstantVelocityTracker(gate_m=3.0, confirm_hits=2)
    stage_ms: Dict[str, List[float]] = {name: [] for name in ("generate", "sync", "fuse", "track")}
    groups = []
    stale_frames = 0

    for index in range(frames):
        t0 = time.perf_counter()
        frame = make_frame(index, provenance)
        stale = index == 2
        bridge.publish_frame(frame, stale_camera=stale)
        stage_ms["generate"].append((time.perf_counter() - t0) * 1000.0)
        if stale:
            stale_frames += 1

        t0 = time.perf_counter()
        for modality, topic in (
            ("camera", "/camera/front/image"),
            ("lidar", "/lidar/points"),
            ("radar", "/radar/tracks"),
            ("imu", "/imu/data"),
        ):
            for sample in graph.take(topic):
                sync.push(sample)
        group = sync.pop_aligned()
        stage_ms["sync"].append((time.perf_counter() - t0) * 1000.0)
        if group is None:
            continue
        groups.append(group)

        features = _features(group)
        scenario = torch.tensor([1])
        t0 = time.perf_counter()
        with torch.no_grad():
            _fused, _weights, valid = model(features, scenario)
        stage_ms["fuse"].append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        detections = []
        if group["lidar"] is not None:
            for actor in frame.actors:
                detections.append(Detection(group["lidar"].stamp_ns, actor.position, actor.actor_id))
        step = tracker.step(detections)
        stage_ms["track"].append((time.perf_counter() - t0) * 1000.0)
        groups[-1] = {
            "sequence": index,
            "camera_valid": group["camera"] is not None,
            "modalities_present": int(valid.sum().item()),
            "live_tracks": len(step.tracks),
            "births": step.births,
            "matches": step.matches,
        }

    fusion_bench = benchmark_callable(lambda: model(_ones(), torch.tensor([0])), warmup=5, iterations=20)
    return {
        "provenance": {
            "simulator": provenance.simulator,
            "simulator_version": provenance.simulator_version,
            "scene": provenance.scene,
            "seed": provenance.seed,
            "weather": provenance.weather,
            "lighting": provenance.lighting,
            "calibration_id": provenance.calibration_id,
            "label_schema": provenance.label_schema,
            "backend": provenance.backend,
        },
        "boundary": (
            "In-process software-in-the-loop timing. Not a sensor-to-actuator "
            "measurement and not an ASIL-D certification."
        ),
        "perception_target": DeploymentTarget.PYTORCH.value,
        "tensorrt_llm_on_perception_path": allows_perception(DeploymentTarget.TENSORRT_LLM),
        "frames": frames,
        "stale_camera_frames": stale_frames,
        "camera_drops_kept_last": graph.dropped.get("/camera/front/image", 0),
        "aligned_groups": len(groups),
        "groups": groups,
        "stage_ms": {name: _summary(values) for name, values in stage_ms.items()},
        "fusion_benchmark_ms": fusion_bench,
    }


def _features(group: Dict):
    features = {}
    for name in FEATURE_DIMS:
        sample = group.get(name)
        if sample is None:
            features[name] = None
        else:
            features[name] = torch.ones(1, FEATURE_DIMS[name])
    return features


def _ones():
    return {name: torch.ones(1, dim) for name, dim in FEATURE_DIMS.items()}


def _summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0.0, "mean_ms": 0.0, "max_ms": 0.0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {"count": float(len(values)), "mean_ms": float(tensor.mean()), "max_ms": float(tensor.max())}


def main() -> None:
    report = run_feedback_loop()
    print(report["boundary"])
    print(f"backend={report['provenance']['backend']} scene={report['provenance']['scene']}")
    print(f"aligned_groups={report['aligned_groups']} stale_camera_frames={report['stale_camera_frames']}")
    for name, stats in report["stage_ms"].items():
        print(f"  {name}: mean {stats['mean_ms']:.3f} ms")
    bench = report["fusion_benchmark_ms"]
    print(
        f"fusion benchmark p50 {bench['p50_ms']:.3f} ms, "
        f"p95 {bench['p95_ms']:.3f} ms over {int(bench['iterations'])} iterations"
    )
