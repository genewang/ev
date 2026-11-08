"""Tests for the sensor contract, sync policy, fusion mask, and timing report."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fusion.context_attention import ContextConditionedAttention
from src.fusion.tracking import ConstantVelocityTracker, Detection
from src.inference.benchmark import benchmark_callable
from src.inference.targets import DeploymentTarget, allows_perception
from src.pipeline.feedback_loop import run_feedback_loop
from src.ros2_integration.graph import InProcessGraph
from src.ros2_integration.messages import CAMERA_QOS, CONTROL_QOS, MAP_QOS, SensorSample
from src.ros2_integration.sync import TimestampSynchronizer


def _sample(modality, topic, stamp_ns):
    return SensorSample(
        topic=topic,
        modality=modality,
        stamp_ns=stamp_ns,
        frame_id="base_link",
        sequence=0,
        source="test",
        valid=True,
    )


def test_camera_qos_keeps_only_the_latest_frame():
    graph = InProcessGraph()
    for i in range(5):
        graph.publish(_sample("camera", "/camera/front/image", i))
    assert graph.depth("/camera/front/image") == CAMERA_QOS.depth
    assert graph.dropped["/camera/front/image"] == 4
    kept = graph.take("/camera/front/image")
    assert kept[0].stamp_ns == 4


def test_control_and_map_qos_differ_from_camera():
    assert CAMERA_QOS.reliability == "BEST_EFFORT"
    assert CONTROL_QOS.reliability == "RELIABLE"
    assert MAP_QOS.durability == "TRANSIENT_LOCAL"


def test_sync_uses_acquisition_time_not_the_latest_sample():
    sync = TimestampSynchronizer(["camera", "lidar"], slop_ns=30_000_000, anchor="lidar")
    sync.push(_sample("camera", "/camera/front/image", 120_000_000))
    sync.push(_sample("camera", "/camera/front/image", 200_000_000))
    sync.push(_sample("lidar", "/lidar/points", 145_000_000))
    group = sync.pop_aligned()
    assert group["camera"].stamp_ns == 120_000_000
    assert group["lidar"].stamp_ns == 145_000_000


def test_sync_leaves_a_far_sample_unaligned():
    sync = TimestampSynchronizer(["camera", "lidar"], slop_ns=10_000_000, anchor="lidar")
    sync.push(_sample("camera", "/camera/front/image", 200_000_000))
    sync.push(_sample("lidar", "/lidar/points", 145_000_000))
    group = sync.pop_aligned()
    assert group["camera"] is None
    assert sync._buffers["camera"][0].stamp_ns == 200_000_000


def test_missing_first_modality_does_not_touch_an_uninitialized_tensor():
    model = ContextConditionedAttention(
        {"lidar": 4, "camera": 4, "radar": 4}, embed_dim=16, num_heads=4
    )
    fused, weights, valid = model(
        {"lidar": None, "camera": torch.ones(2, 4), "radar": torch.ones(2, 4)},
        torch.tensor([0, 1]),
    )
    assert fused.shape == (2, 16)
    assert weights.shape[0] == 2
    assert valid[:, 0].sum().item() == 0
    assert not torch.isnan(fused).any()


def test_tracker_confirms_then_drops_a_lost_object():
    tracker = ConstantVelocityTracker(gate_m=1.0, confirm_hits=2, max_misses=1)
    first = tracker.step([Detection(0, (0.0, 0.0, 0.0), "ped")])
    second = tracker.step([Detection(1_000_000_000, (0.4, 0.0, 0.0), "ped")])
    assert first.births == 1
    assert second.matches == 1
    assert second.tracks[0].confirmed
    lost = tracker.step([])
    assert lost.losses == 0
    lost = tracker.step([])
    assert lost.losses == 1


def test_benchmark_reports_percentiles():
    stats = benchmark_callable(lambda: torch.zeros(8), warmup=2, iterations=10)
    assert stats["iterations"] == 10
    assert stats["p50_ms"] <= stats["p95_ms"] <= stats["p99_ms"] <= stats["max_ms"]


def test_tensorrt_llm_is_not_a_perception_target():
    assert allows_perception(DeploymentTarget.TENSORRT)
    assert allows_perception(DeploymentTarget.COREML_ANE)
    assert not allows_perception(DeploymentTarget.TENSORRT_LLM)


def test_feedback_loop_records_provenance_and_a_stale_camera():
    report = run_feedback_loop(frames=4, seed=3)
    assert report["provenance"]["backend"] == "stand-in"
    assert report["provenance"]["simulator"] == "isaac-stand-in"
    assert report["provenance"]["seed"] == 3
    assert report["tensorrt_llm_on_perception_path"] is False
    assert report["aligned_groups"] == 4
    assert report["groups"][2]["camera_valid"] is False
    assert report["groups"][0]["camera_valid"] is True
    assert report["groups"][3]["live_tracks"] >= 1
    assert "p50_ms" in report["fusion_benchmark_ms"]
    assert "ASIL-D certification" in report["boundary"]
