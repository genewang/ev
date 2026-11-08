# Benchmark methodology

`src/inference/benchmark.py` measures one callable.

1. Warm up the callable before the timed window.
2. On CUDA, time with CUDA events and synchronize. On CPU or Apple MPS, use a clock around a device synchronize when that device exists.
3. Report mean, p50, p95, p99, and max. A single `time.time()` sample is not a budget.
4. Say what the callable includes. Model forward is not sensor-to-output latency.

The 32 ms figure in older notes is a target for a future vehicle measurement. The feedback-loop report labels its numbers as in-process software-in-the-loop timing.

Run:

```bash
python benchmarks/run_feedback_loop.py
python -m pytest tests/test_feedback_loop.py tests/test_perception.py::TestPointNetPlusPlus -q
```
