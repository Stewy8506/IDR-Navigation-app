import numpy as np
import pandas as pd
import torch

from ml.src.model import SpeedVibrationFilterNet

df_s = pd.read_csv("ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3a/S-S3a.csv", encoding="latin1").iloc[:5000]
df_s.columns = df_s.columns.str.strip()

gt = df_s["Ground_Truth_Speed_mps"].values

print("GT speed at k=1200 to 2100:", gt[1200:2100].mean())
