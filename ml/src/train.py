"""
train.py - Training script for SpeedVibrationFilterNet on IO-VNBD dataset.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .model import SpeedVibrationFilterNet
from .dataset import IOVNBDDataset


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train Speed & Vibration ML Model")
    parser.add_argument("--data_dir", type=str, default="ml/data/IO-VNBD", help="Path to IO-VNBD dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--window_size", type=int, default=100, help="Window size in samples")
    parser.add_argument("--output_dir", type=str, default="ml/weights", help="Directory to save checkpoints")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SpeedVibrationFilterNet(in_channels=6, window_size=args.window_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_dataset = IOVNBDDataset(data_dir=args.data_dir, window_size=args.window_size, is_train=True)
    if len(train_dataset) == 0:
        print(f"No training data found in {args.data_dir}. Please place dataset files in that directory.")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Starting training on {len(train_dataset)} windows...")
    
    best_loss = float("inf")
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {loss:.4f}")
        
        if loss < best_loss:
            best_loss = loss
            save_path = os.path.join(args.output_dir, "best_speed_filter.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")


if __name__ == "__main__":
    main()
