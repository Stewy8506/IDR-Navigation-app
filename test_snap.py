import numpy as np
import pandas as pd
import sys
import math

from ml.src.map_matcher import OsmRoadGraph, HmmMapMatcher
from ml.src.evaluate_full_pipeline import run_ekf

df_s = pd.read_csv("ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw11/S-Vw11.csv", encoding="latin1").iloc[:4909]
gt_enu = np.column_stack((df_s["Ground_Truth_East"], df_s["Ground_Truth_North"]))

graph = OsmRoadGraph()
graph.load_from_waypoints(gt_enu[:, :2])
print("Graph loaded with", len(graph.segments), "segments")
