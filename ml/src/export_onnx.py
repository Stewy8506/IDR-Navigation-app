"""
export_onnx.py - Exports trained 16-channel Spectral Model to ONNX Runtime format.
"""

import os
import torch
from .model import SpeedVibrationFilterNet


def main():
    weights_path = "ml/weights/best_spectral_speed_filter.pt"
    onnx_path = "app/assets/models/speed_filter.onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

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
    print(f"Successfully exported 16-channel model to {onnx_path} (Size: {os.path.getsize(onnx_path)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
