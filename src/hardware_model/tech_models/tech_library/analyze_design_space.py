#filename = "/nfs/pool0/pmcewen/rsgvm19dir/codesign/src/hardware_model/tech_models/tech_library/design_spaces/tech_sweep_mvs_1_spice_cfg_20260129_184108.csv"

import pandas as pd
import sys
def analyze_design_space(filename):
    df = pd.read_csv(filename)

    print(df.columns)
    #mask = (df["V_dd"] >= 1) & (df["delay"] <= 0.1) & (df["V_th_eff"] >= 0.2)
    mask = (df["V_dd"] <= 1.28) & (df["delay"] <= 0.02) & (df["V_th_eff"] >= 0.1)
    df_filtered = df[mask]
    cols_to_show = ["L", "W", "V_dd","V_th_eff", "tox", "delay","Ieff", "Ioff"]
    #cols_to_show = ["delay", "Edynamic", "Pstatic", "area", "slope_at_crossing", "NM_H", "NM_L"]
    sorted_df = df_filtered.sort_values(by="delay", ascending=True)
    pd.set_option('display.max_rows', None)
    print(sorted_df[cols_to_show])

if __name__ == "__main__":
    args = sys.argv[1:]
    filename = args[0]
    analyze_design_space(filename)