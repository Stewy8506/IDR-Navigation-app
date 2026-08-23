"""
export_tflite.py - Exports 10-Channel Dual-Head PyTorch model to ONNX and creates metadata JSON.
"""

import argparse
import json
import os
import torch

from .model import SpeedVibrationFilterNet


def export_to_onnx(model, onnx_path: str, in_channels: int = 10, window_size: int = 20):
    model.eval()
    dummy_input = torch.randn(1, in_channels, window_size, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["imu_input"],
        output_names=["speed_and_variance_output"],
    )
    print(f"Exported ONNX model to {onnx_path}")


def save_metadata(
    meta_path: str, in_channels: int = 10, window_size: int = 20, sampling_rate_hz: int = 10
):
    meta = {
        "model_name": "SpeedVibrationFilterNet_Physics10Ch",
        "version": "3.0",
        "in_channels": in_channels,
        "input_channels": [
            "accel_x_mps2",
            "accel_y_mps2",
            "accel_z_mps2",
            "gyro_yaw_rads",
            "gyro_pitch_rads",
            "gyro_roll_rads",
            "accel_norm_dynamic_mps2",
            "gyro_norm_rads",
            "cumulative_velocity_integral_mps",
            "vertical_vibration_energy_mps4",
        ],
        "window_size": window_size,
        "sampling_rate_hz": sampling_rate_hz,
        "duration_seconds": window_size / sampling_rate_hz,
        "input_shape": [1, in_channels, window_size],
        "output_shape": [1, 2],
        "outputs": [
            {"name": "speed_mps", "index": 0, "unit": "m/s", "description": "Estimated forward vehicle speed"},
            {
                "name": "variance_mps2",
                "index": 1,
                "unit": "(m/s)^2",
                "description": "Measurement noise covariance R for EKF fusion",
            },
        ],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="ml/weights/best_speed_filter.pt")
    parser.add_argument("--output_dir", type=str, default="app/assets/models")
    parser.add_argument("--in_channels", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = SpeedVibrationFilterNet(in_channels=args.in_channels, window_size=args.window_size)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        print(f"Loaded weights from {args.weights}")
    else:
        print(f"Note: Exporting initialized model (weights not found at {args.weights})")

    onnx_path = os.path.join(args.output_dir, "speed_filter.onnx")
    meta_path = os.path.join(args.output_dir, "model_metadata.json")

    export_to_onnx(model, onnx_path, in_channels=args.in_channels, window_size=args.window_size)
    save_metadata(meta_path, in_channels=args.in_channels, window_size=args.window_size)


if __name__ == "__main__":
    main()
