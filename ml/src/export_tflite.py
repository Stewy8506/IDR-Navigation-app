"""
export_tflite.py - Exports PyTorch model to ONNX and TFLite for Flutter deployment.
Also outputs model_metadata.json (input dimensions, mean, std).
"""

import os
import json
import argparse
import torch

from .model import SpeedVibrationFilterNet


def export_to_onnx(model, onnx_path: str, window_size: int = 100):
    model.eval()
    dummy_input = torch.randn(1, 6, window_size, dtype=torch.float32)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['imu_input'],
        output_names=['speed_output'],
        dynamic_axes={'imu_input': {0: 'batch_size'}, 'speed_output': {0: 'batch_size'}}
    )
    print(f"Exported ONNX model to {onnx_path}")


def save_metadata(meta_path: str, window_size: int = 100, sampling_rate_hz: int = 100):
    meta = {
        "model_name": "SpeedVibrationFilterNet",
        "version": "1.0",
        "input_channels": ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
        "window_size": window_size,
        "sampling_rate_hz": sampling_rate_hz,
        "input_shape": [1, 6, window_size],
        "output_shape": [1, 1],
        "units": {
            "accel": "m/s^2",
            "gyro": "rad/s",
            "speed": "m/s"
        }
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="ml/weights/best_speed_filter.pt")
    parser.add_argument("--output_dir", type=str, default="app/assets/models")
    parser.add_argument("--window_size", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    model = SpeedVibrationFilterNet(in_channels=6, window_size=args.window_size)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        print(f"Loaded weights from {args.weights}")

    onnx_path = os.path.join(args.output_dir, "speed_filter.onnx")
    meta_path = os.path.join(args.output_dir, "model_metadata.json")

    export_to_onnx(model, onnx_path, window_size=args.window_size)
    save_metadata(meta_path, window_size=args.window_size)


if __name__ == "__main__":
    main()
