import numpy as np
import pandas as pd
import torch

from ml.src.model import SpeedVibrationFilterNet
from ml.src.dataset_spectral import compute_spectral_physics_features

df_s = pd.read_csv("ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3a/S-S3a.csv", encoding="latin1").iloc[:5000]
df_v = pd.read_csv("ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3a/V-S3a.csv", encoding="latin1").iloc[:5000]
df_s.columns = df_s.columns.str.strip()
df_v.columns = df_v.columns.str.strip()

speed_model = SpeedVibrationFilterNet(in_channels=16)
speed_model.load_state_dict(torch.load("ml/weights/best_spectral_speed_filter.pt", map_location="cpu", weights_only=True))
speed_model.eval()

ai_speed_mps = np.zeros(5000)
for k in range(30, 5000):
    window = df_v.iloc[k-30:k]
    x_tensor = compute_spectral_physics_features(window)
    with torch.no_grad():
        speed_pred, _ = speed_model(x_tensor)
        ai_speed_mps[k] = speed_pred.item()

outage_start = 1200
outage_end = 2100

print("Average GT Speed in outage:", df_s["Ground_Truth_Speed_mps"].iloc[outage_start:outage_end].mean())
print("Average AI Speed in outage:", ai_speed_mps[outage_start:outage_end].mean())

