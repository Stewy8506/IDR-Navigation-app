import numpy as np
import pandas as pd
import math

df_s = pd.read_csv("ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw11/S-Vw11.csv", encoding="latin1").iloc[:4909]
df_s.columns = df_s.columns.str.strip()
gt_enu = np.column_stack((df_s["Math_ENU_East"], df_s["Math_ENU_North"]))

step = 100
for i in range(1200, 2000, step):
    start = gt_enu[i]
    end = gt_enu[i + step]
    de = end[0] - start[0]
    dn = end[1] - start[1]
    heading = math.atan2(dn, de)
    print(f"Index {i}, length={math.hypot(de, dn):.2f}, heading={math.degrees(heading):.1f}")
