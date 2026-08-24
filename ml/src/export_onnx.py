"""
export_onnx.py - Exports DeepSpeedKinematicsNet and DeepHeadingObserverNet to ONNX format.
"""

import os
import torch
from .model import DeepSpeedKinematicsNet, DeepHeadingObserverNet


class SpeedExportWrapper(torch.nn.Module):
    """
    Wrapper exposing deterministic (B, 5) output tensor:
    [mu_v, var_v, delta_v, p_zupt, pitch]
    """
    def __init__(self, model: DeepSpeedKinematicsNet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return torch.stack([
            out["mu_v"],
            out["var_v"],
            out["delta_v"],
            out["p_zupt"],
            out["pitch"],
        ], dim=-1)


class HeadingExportWrapper(torch.nn.Module):
    """
    Wrapper exposing deterministic (B, 3) output tensor:
    [gyro_bias_z, delta_wz, bias_var]
    """
    def __init__(self, model: DeepHeadingObserverNet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return torch.stack([
            out["gyro_bias_z"],
            out["delta_wz"],
            out["bias_var"],
        ], dim=-1)


def export_speed_model():
    weights_path = "ml/weights/best_spectral_speed_filter.pt"
    onnx_path = "app/assets/models/speed_filter.onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    if not os.path.exists(weights_path):
        print(f"Warning: {weights_path} not found.")
        return

    core = DeepSpeedKinematicsNet(in_channels=18, window_size=48)
    core.load_state_dict(torch.load(weights_path, map_location="cpu"))
    core.eval()

    model = SpeedExportWrapper(core)
    model.eval()

    dummy_input = torch.randn(1, 18, 48, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["imu_physics_18ch"],
        output_names=["speed_kinematics_5d"],
        dynamic_axes={"imu_physics_18ch": {0: "batch_size"}, "speed_kinematics_5d": {0: "batch_size"}},
    )
    print(f"Successfully exported DeepSpeedKinematicsNet to {onnx_path} (Size: {os.path.getsize(onnx_path)/1024:.1f} KB)")


def export_heading_model():
    weights_path = "ml/weights/best_heading_observer.pt"
    onnx_path = "app/assets/models/heading_filter.onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    if not os.path.exists(weights_path):
        print(f"Warning: {weights_path} not found.")
        return

    core = DeepHeadingObserverNet(in_channels=6)
    core.load_state_dict(torch.load(weights_path, map_location="cpu"))
    core.eval()

    model = HeadingExportWrapper(core)
    model.eval()

    dummy_input = torch.randn(1, 6, 48, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["imu_raw_6ch"],
        output_names=["heading_kinematics_3d"],
        dynamic_axes={"imu_raw_6ch": {0: "batch_size"}, "heading_kinematics_3d": {0: "batch_size"}},
    )
    print(f"Successfully exported DeepHeadingObserverNet to {onnx_path} (Size: {os.path.getsize(onnx_path)/1024:.1f} KB)")


def main():
    export_speed_model()
    export_heading_model()


if __name__ == "__main__":
    main()
