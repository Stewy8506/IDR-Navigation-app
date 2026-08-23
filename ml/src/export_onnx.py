"""
export_onnx.py - Exports trained 16-channel Spectral / Recurrent Model to ONNX Runtime format.
"""

import os
import torch
from .model import SpeedVibrationFilterNet, RecurrentSpeedFilterNet


def export_spectral():
    weights_path = "ml/weights/best_spectral_speed_filter.pt"
    onnx_path = "app/assets/models/speed_filter.onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    if not os.path.exists(weights_path):
        return

    model = SpeedVibrationFilterNet(in_channels=16, window_size=32)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 16, 32, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["imu_spectral_16ch"],
        output_names=["speed_and_variance"],
        dynamic_axes={"imu_spectral_16ch": {0: "batch_size"}, "speed_and_variance": {0: "batch_size"}},
    )
    print(f"Successfully exported Spectral model to {onnx_path} (Size: {os.path.getsize(onnx_path)/1024:.1f} KB)")


def export_recurrent():
    weights_path = "ml/weights/best_recurrent_speed_filter.pt"
    onnx_path = "app/assets/models/recurrent_speed_filter.onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    if not os.path.exists(weights_path):
        return

    model = RecurrentSpeedFilterNet(in_channels=16, window_size=32, hidden_dim=128, num_layers=2, use_prior_speed=True)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 16, 32, dtype=torch.float32)
    dummy_v_prior = torch.zeros(1, 1, dtype=torch.float32)
    dummy_h = torch.zeros(2, 1, 128, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_input, dummy_v_prior, dummy_h),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["imu_spectral_16ch", "v_prior", "hidden_state_in"],
        output_names=["speed_and_variance", "hidden_state_out"],
        dynamic_axes={
            "imu_spectral_16ch": {0: "batch_size"},
            "v_prior": {0: "batch_size"},
            "hidden_state_in": {1: "batch_size"},
            "speed_and_variance": {0: "batch_size"},
            "hidden_state_out": {1: "batch_size"},
        },
    )
    print(f"Successfully exported Prior-Conditioned Recurrent model to {onnx_path} (Size: {os.path.getsize(onnx_path)/1024:.1f} KB)")


def main():
    export_spectral()
    export_recurrent()


if __name__ == "__main__":
    main()

