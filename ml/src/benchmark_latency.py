"""
benchmark_latency.py - High-Precision CPU & ONNX Runtime Latency Benchmark (Task 2).
Measures:
  1. DeepSpeedKinematicsNet (18-Ch x 48-Step ConvNeXt-1D + Self-Attention)
  2. DeepHeadingObserverNet (6-Ch x 48-Step TCN)
Performs 200 warmup iterations followed by 1,000 timed runs.
Reports: Mean, Median (P50), P90, P95, P99, Min, Max.
"""

import os
import time
import numpy as np
import torch
import onnxruntime as ort

from .model import DeepSpeedKinematicsNet, DeepHeadingObserverNet


def benchmark_pytorch_cpu(model: torch.nn.Module, dummy_input: torch.Tensor, name: str, runs: int = 1000, warmup: int = 200) -> dict:
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
            
    # Timed runs
    latencies_ms = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
            
    latencies_ms = np.array(latencies_ms)
    return {
        "name": name,
        "runtime": "PyTorch CPU",
        "runs": runs,
        "mean": float(np.mean(latencies_ms)),
        "median": float(np.median(latencies_ms)),
        "p90": float(np.percentile(latencies_ms, 90)),
        "p95": float(np.percentile(latencies_ms, 95)),
        "p99": float(np.percentile(latencies_ms, 99)),
        "min": float(np.min(latencies_ms)),
        "max": float(np.max(latencies_ms)),
    }


def benchmark_onnx(onnx_path: str, dummy_input: np.ndarray, input_name: str, name: str, runs: int = 1000, warmup: int = 200) -> dict:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
    feed = {input_name: dummy_input}

    # Warmup
    for _ in range(warmup):
        _ = session.run(None, feed)

    # Timed runs
    latencies_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = session.run(None, feed)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms = np.array(latencies_ms)
    return {
        "name": name,
        "runtime": "ONNX Runtime (Single-Thread CPU)",
        "runs": runs,
        "mean": float(np.mean(latencies_ms)),
        "median": float(np.median(latencies_ms)),
        "p90": float(np.percentile(latencies_ms, 90)),
        "p95": float(np.percentile(latencies_ms, 95)),
        "p99": float(np.percentile(latencies_ms, 99)),
        "min": float(np.min(latencies_ms)),
        "max": float(np.max(latencies_ms)),
    }


def print_report(res: dict):
    print(f"\nModel:   {res['name']} ({res['runtime']})")
    print(f"Runs:    {res['runs']:,} iterations (after 200 warmup runs)")
    print(f"-------------------------------------------------------------")
    print(f"  Mean:        {res['mean']:.4f} ms ({res['mean']*1000:.1f} µs)")
    print(f"  Median/P50:  {res['median']:.4f} ms ({res['median']*1000:.1f} µs)")
    print(f"  P90:         {res['p90']:.4f} ms ({res['p90']*1000:.1f} µs)")
    print(f"  P95:         {res['p95']:.4f} ms ({res['p95']*1000:.1f} µs)")
    print(f"  P99:         {res['p99']:.4f} ms ({res['p99']*1000:.1f} µs)")
    print(f"  Min / Max:   {res['min']:.4f} ms / {res['max']:.4f} ms")
    print(f"-------------------------------------------------------------")
    passed = res['p99'] < 20.0
    pref_passed = res['p95'] < 5.0
    print(f"  Target Budget: < 20.0 ms | P99 Status: {'[PASS]' if passed else '[FAIL]'}")
    print(f"  Preferred:     < 5.0 ms  | P95 Status: {'[PASS]' if pref_passed else '[WARN]'}")


def main():
    print("=" * 65)
    print("   TASK 2: HIGH-CAPACITY AI MODEL INFERENCE LATENCY BENCHMARK")
    print("=" * 65)

    speed_model = DeepSpeedKinematicsNet(in_channels=18, window_size=48)
    head_model = DeepHeadingObserverNet(in_channels=6)

    dummy_speed = torch.randn(1, 18, 48, dtype=torch.float32)
    dummy_head = torch.randn(1, 6, 48, dtype=torch.float32)

    # 1. PyTorch CPU
    res_speed_pt = benchmark_pytorch_cpu(speed_model, dummy_speed, "DeepSpeedKinematicsNet (~1.2M Params)")
    print_report(res_speed_pt)

    res_head_pt = benchmark_pytorch_cpu(head_model, dummy_head, "DeepHeadingObserverNet (~385K Params)")
    print_report(res_head_pt)

    # 2. ONNX Runtime (if exported)
    onnx_speed_path = "app/assets/models/speed_filter.onnx"
    if os.path.exists(onnx_speed_path):
        res_speed_onnx = benchmark_onnx(onnx_speed_path, dummy_speed.numpy(), "imu_physics_18ch", "DeepSpeedKinematicsNet (ONNX)")
        print_report(res_speed_onnx)

    onnx_head_path = "app/assets/models/heading_filter.onnx"
    if os.path.exists(onnx_head_path):
        res_head_onnx = benchmark_onnx(onnx_head_path, dummy_head.numpy(), "imu_raw_6ch", "DeepHeadingObserverNet (ONNX)")
        print_report(res_head_onnx)


if __name__ == "__main__":
    main()
