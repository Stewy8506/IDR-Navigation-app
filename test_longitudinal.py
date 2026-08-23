import numpy as np
import pandas as pd
import sys

from ml.src.map_matcher import OsmRoadGraph, HmmMapMatcher
from ml.src.evaluate_full_pipeline import run_ekf

df_s = pd.read_csv("ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw11/S-Vw11.csv", encoding="latin1").iloc[:4909]
df_s.columns = df_s.columns.str.strip()
df_v = pd.read_csv("ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw11/V-Vw11.csv", encoding="latin1").iloc[:4909]
df_v.columns = df_v.columns.str.strip()

gt_enu = np.column_stack((df_s["Math_ENU_East"], df_s["Math_ENU_North"]))
pos_outage_ai = np.load("ml/evaluation_plots/trajectory_drift_benchmark.png") # Not available
