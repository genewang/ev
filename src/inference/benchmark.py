"""
Latency measurement for a callable.

A single ``time.time()`` sample is not a latency budget. This helper warms up,
synchronizes CUDA when that device is used, and reports mean and tail percentiles.
The numbers describe the callable that was passed in, not a full vehicle path,
unless the callable itself includes that path.
"""

import time
from typing import Callable, Dict

import torch


def benchmark_callable(
    fn: Callable[[], None],
    warmup: int = 10,
    iterations: int = 50,
    device: torch.device = None,
) -> Dict[str, float]:
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be >= 0 and iterations must be >= 1")
    device = device or torch.device("cpu")
    for _ in range(warmup):
        fn()
    samples = _sample(fn, iterations, device)
    values = torch.tensor(samples, dtype=torch.float64)
    return {
        "iterations": float(iterations),
        "mean_ms": float(values.mean()),
        "p50_ms": float(torch.quantile(values, 0.50)),
        "p95_ms": float(torch.quantile(values, 0.95)),
        "p99_ms": float(torch.quantile(values, 0.99)),
        "max_ms": float(values.max()),
    }


def _sample(fn: Callable[[], None], iterations: int, device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(iterations):
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        return samples

    samples = []
    for _ in range(iterations):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def _sync(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
