filename = "/nfs/pool0/pmcewen/rsgvm19dir/codesign/src/hardware_model/tech_models/tech_library/design_spaces/tech_sweep_mvs_self_consistent_cfg_20260128_135657.csv"

import pandas as pd

df = pd.read_csv(filename)

#mask = (df["V_dd"] >= 1) & (df["delay"] <= 0.1) & (df["V_th_eff"] >= 0.2)
mask = (df["delay"] <= 0.02)
df_filtered = df[mask]
cols_to_show = ["L", "V_dd", "V_th", "V_th_eff", "tox", "tsemi", "Lscale", "dVt", "delta", "n0", "k_gate", "delay", "Edynamic", "Pstatic", "Ieff", "Ioff", "slope_at_crossing", "NM_H", "NM_L"]
sorted_df = df_filtered.sort_values(by="delay", ascending=True)
pd.set_option('display.max_rows', None)
print(sorted_df[cols_to_show])