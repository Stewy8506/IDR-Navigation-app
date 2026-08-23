"""
evaluate.py - Benchmark evaluation script for speed estimation and drift metric calculations.
"""

import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from .model import SpeedVibrationFilterNet
from .dataset import IOVNBDDataset


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze(-1)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.numpy())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mae = np.mean(np.abs(preds - targets))
    return rmse, mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="ml/data/IO-VNBD")
    parser.add_argument("--weights", type=str, default="ml/weights/best_speed_filter.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeedVibrationFilterNet().to(device)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights from {args.weights}")

    test_dataset = IOVNBDDataset(data_dir=args.data_dir, is_train=False)
    if len(test_dataset) == 0:
        print(f"No test data in {args.data_dir}")
        return

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    rmse, mae = evaluate(model, test_loader, device)
    print(f"Evaluation Results:\nRMSE: {rmse:.4f} m/s\nMAE:  {mae:.4f} m/s")


if __name__ == "__main__":
    main()
