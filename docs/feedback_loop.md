# Production feedback loop

The production loop connects the truck, ROS 2, Isaac Sim, the full PointNet++ perception stack, and TensorRT. `benchmarks/run_feedback_loop.py` runs those same message contracts in process: publish, acquisition-time sync, masked fusion, tracking, and percentile timing. A live Isaac Sim ROS 2 bridge and a vehicle TensorRT engine attach at this boundary without changing the perception nodes.

```
scenario stand-in
    camera, lidar, radar, imu
        |
        v
in-process graph  (QoS depth, drop count)
        |
        v
acquisition-time sync  (slop window, not "latest message")
        |
        v
context-conditioned attention  (validity mask, no language model)
        |
        v
constant-velocity tracker
        |
        v
stage times and a warmed-up percentile benchmark
```

## What each piece is

| Piece | In this repository | Not in this repository |
| --- | --- | --- |
| ROS 2 | Topic names, QoS profiles, bounded queues, timestamp sync | A running DDS graph. Install ROS 2 before using `rclpy`. |
| Isaac Sim | `IsaacRosBridge` publishes the same topics and records seed, weather, calibration, and label schema | Omniverse, USD scenes, or a real ROS 2 bridge process |
| Multimodal fusion | Masked attention over lidar, camera, radar, and imu embeddings | Trained encoders or a measured detection gain |
| Inference | Percentile timing of the callable you pass, plus separate deployment targets | A TensorRT engine or a 32 ms end-to-end proof |
| Safety | Timeouts and fault flags in the safety package | An ISO 26262 ASIL-D safety case |

## QoS starting points

| Stream | Profile | Reason |
| --- | --- | --- |
| `/camera/front/image` | Best effort, keep last, depth 1 | Prefer the newest frame |
| `/lidar/points` | Reliable, keep last, depth 5 | Losing a scan hurts tracking, but the queue stays bounded |
| `/vehicle/command` | Reliable, keep last, depth 10 | A dropped command is not treated like a stale image |
| `/map/static` | Reliable, transient local | A late subscriber can still read the current map |

These are design starting points. A vehicle still has to measure loss, latency, and incompatible QoS pairs.

## Synchronization

A camera sample at 120 ms and a lidar sample at 145 ms match under a 30 ms slop. A camera sample at 200 ms does not replace the 120 ms sample, and it is not fused when the slop is 10 ms. The far sample stays in the buffer.

## Deployment targets

Perception may move from PyTorch to ONNX Runtime, TensorRT on an NVIDIA GPU, or Core ML on Apple Silicon. Those are different builds. TensorRT-LLM is only for language-model serving and is rejected by `allows_perception`.

## Provenance

Every simulated frame carries simulator name, version, scene, seed, weather, lighting, calibration id, and label schema. The stand-in version is `"0"` until a real Isaac Sim session fills those fields.
