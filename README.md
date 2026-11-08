# EV Autonomous Trucking Perception System

Production perception stack for autonomous electric trucks. Full PointNet++ on the LiDAR path, multimodal fusion across LiDAR, camera, and radar, a ROS 2 execution layer, NVIDIA Isaac Sim for software-in-the-loop, and TensorRT inference inside a 32 ms vehicle budget.

## Project overview

- **Full PointNet++** feature extraction: farthest-point sampling, ball query, and three set-abstraction levels (512, 128, global)
- **Multimodal fusion** of 4× 128-line LiDAR, 8× 8MP cameras, and 6× 77 GHz radars, with attention conditioned on the driving scenario
- **Neural Engine and TensorRT** acceleration. TensorRT builds the perception engine. TensorRT-LLM serves the language plane outside the fusion slice
- **ASIL-D safety runtime** with deterministic execution, WCET analysis, and fault injection
- **ROS 2** sensor graph: QoS, acquisition-time sync, `tf2` frames, and bounded queues
- **Isaac Sim** scenarios published on the same ROS 2 topics as the vehicle, with labels, calibration, and scenario provenance
- **Fleet learning** with differential privacy, Docker on AWS ECS, and MLflow

## Production-scale runtime

```
Cameras · LiDAR · Radar · IMU · GPS
        |
        v
ROS 2 / DDS
  drivers · timestamps · QoS · tf2 · bounded queues
        |
        +---- Isaac Sim ROS 2 bridge (same message contracts)
        |
        v
Full PointNet++  +  camera / radar encoders
        |
        v
Temporal alignment and multimodal attention
        |
        v
Detection · tracking · occupancy
        |
        v
TensorRT engine on the vehicle GPU
        |
        v
ASIL-D runtime  →  control
        |
        v
Fleet logs, Isaac scenario regen, retraining, engine rebuild
```

| Layer | Production role |
| --- | --- |
| ROS 2 | Distributed nodes, DDS discovery, and per-stream QoS. Camera frames are best-effort keep-last. Control commands are reliable. Maps are transient-local. |
| Isaac Sim | Repeatable scenes, domain randomization, and ground-truth labels. The perception nodes subscribe to the same topics in simulation, replay, and on the truck. |
| PointNet++ | Hierarchical set abstraction on the LiDAR cloud. Level 1: 512 points, radius 0.2, 32 neighbors. Level 2: 128 points, radius 0.4, 64 neighbors. Level 3: global 1024-d feature and a 20-class head. |
| Fusion | Camera, LiDAR, radar, and IMU in one embedding. Missing sensors are masked. Tracking associates objects across frames in a consistent coordinate frame. |
| Inference | Profile the full path, then TensorRT on NVIDIA and the Neural Engine on Apple Silicon development hosts. FP16 and INT8 stay inside the 32 ms budget. Language models use TensorRT-LLM off that path. |
| Safety | ASIL-D runtime, WCET bounds, and fault injection for stale, missing, and corrupt sensor inputs. |

Run the in-process loop that exercises those contracts:

```bash
python benchmarks/run_feedback_loop.py
```

## Performance targets

- **End-to-end latency**: 32 ms, against an 85 ms NVIDIA reference
- **Training efficiency**: 3× fewer labeled samples than voxel methods
- **Throughput**: 4× with TensorRT
- **Scenario design**: 40% less manual effort through Isaac Sim and generated edge cases

Percentile measurement for a callable is `src/inference/benchmark.py`. The vehicle number is the full sensor-to-output path on the target GPU.

## Quick start

```bash
git clone https://github.com/genewang/ev.git
cd ev
pip install -r requirements.txt

python src/main.py
python src/simulation/run_sim.py
python benchmarks/run_feedback_loop.py
python -m pytest tests/ -q
```

## Project structure

```
ev/
├── src/
│   ├── perception/          # Full PointNet++, detection, segmentation
│   ├── fusion/              # Multimodal attention, temporal fusion, tracking
│   ├── ros2_integration/    # QoS, timestamp sync, ROS 2 interface
│   ├── simulation/          # Isaac ROS bridge and synthetic scenes
│   ├── inference/           # TensorRT / Neural Engine targets and benchmarks
│   ├── pipeline/            # Closed loop from sensors to measured stages
│   ├── safety/              # ASIL-D runtime, WCET, fault injection
│   └── ai_tools/            # Ticket triage and language tooling
├── benchmarks/              # Production-loop runner
├── configs/
├── tests/
├── docker/
├── docs/
└── EV_Autonomous_Driving_Demo.ipynb
```

## Dependencies

- Python 3.9+
- PyTorch 2.0+
- Full PointNet++ (`src/perception/pointnet.py`)
- TensorRT for perception, TensorRT-LLM for the language plane
- ROS 2
- NVIDIA Isaac Sim for the ROS 2 bridge
- Docker and the AWS SDK

## Documentation

- [System architecture](docs/architecture.md)
- [Production feedback loop](docs/feedback_loop.md)
- [Benchmark methodology](docs/benchmark_methodology.md)
- [Deployment guide](docs/deployment.md)
- [Safety framework](docs/safety.md)

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the code of conduct and the pull-request process.

## License

MIT. See [LICENSE](LICENSE).
