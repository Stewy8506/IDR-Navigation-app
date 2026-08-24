"""
benchmark_latency.py - Measures on-device CPU inference latency (Mean, P95, P99, Max)
to ensure model stays well within the 100ms real-time budget.
"""

import time
import argparse
import numpy as np
import torch
import onnxruntime as ort

from .model import SpeedVibrationFilterNet


def benchmark_pytorch(model, in_channels=10, window_size=20, iterations=1000):
    model.eval()
    dummy_input = torch.randn(1, in_channels, window_size, dtype=torch.float32)

    # Warmup
    for _ in range(50):
        with torch.no_grad():
            _ = model(dummy_input)

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    return {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "max_ms": np.max(latencies),
    }


def benchmark_onnx(onnx_path, in_channels=10, window_size=20, iterations=1000):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, in_channels, window_size).astype(np.float32)

    # Warmup
    for _ in range(50):
        _ = session.run(None, {input_name: dummy_input})

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    return {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "max_ms": np.max(latencies),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str, default="app/assets/models/speed_filter.onnx")
    parser.add_argument("--in_channels", type=int, default=16)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    print("=" * 55)
    print("      INFERENCE CPU LATENCY BENCHMARK (1,000 RUNS)")
    print("=" * 55)

    # PyTorch CPU Benchmark
    model = SpeedVibrationFilterNet(in_channels=args.in_channels, window_size=args.window_size)
    pt_stats = benchmark_pytorch(
        model, in_channels=args.in_channels, window_size=args.window_size, iterations=args.iterations
    )

    print("\n[PyTorch Single-Threaded CPU]")
    print(f"  Mean Latency: {pt_stats['mean_ms']:.3f} ms ± {pt_stats['std_ms']:.3f} ms")
    print(f"  P50 / Median: {pt_stats['p50_ms']:.3f} ms")
    print(f"  P95:          {pt_stats['p95_ms']:.3f} ms")
    print(f"  P99:          {pt_stats['p99_ms']:.3f} ms")

    # ONNX Runtime CPU Benchmark
    try:
        onnx_stats = benchmark_onnx(
            args.onnx_path,
            in_channels=args.in_channels,
            window_size=args.window_size,
            iterations=args.iterations,
        )
        print("\n[ONNX Runtime Mobile/CPU Provider]")
        print(f"  Mean Latency: {onnx_stats['mean_ms']:.3f} ms ± {onnx_stats['std_ms']:.3f} ms")
        print(f"  P50 / Median: {onnx_stats['p50_ms']:.3f} ms")
        print(f"  P95:          {onnx_stats['p95_ms']:.3f} ms")
        print(f"  P99:          {onnx_stats['p99_ms']:.3f} ms")
    except Exception as e:
        print(f"\nONNX benchmark skipped: {e}")

    print("\n" + "=" * 55)
    print(f"Target 10 Hz Budget:  100.0 ms")
    print(f"Margin Remaining:     >{100.0 - pt_stats['p99_ms']:.2f} ms")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
