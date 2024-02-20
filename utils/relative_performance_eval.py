import pandas as pd
import numpy as np
from utility import *

df_performance_fullyear = pd.read_csv("utils/df_performance_fullyear.csv")
df_performance_halfyear = pd.read_csv("utils/df_performance_halfyear.csv")

lst_etf_name = [
    "FUEMAV30", "FUESSV30", "FUESSV50", "FUEVFVND", "FUEVN100", "VNI"
]

fuemav30_halfyear = pd.read_csv("data/benchmark_performance/FUEMAV30_HALFYEAR.csv")
fuessv30_halfyear = pd.read_csv("data/benchmark_performance/FUESSV30_HALFYEAR.csv")
fuessv50_halfyear = pd.read_csv("data/benchmark_performance/FUESSV50_HALFYEAR.csv")
fuevfvnd_halfyear = pd.read_csv("data/benchmark_performance/FUEVFVND_HALFYEAR.csv")
fuevn100_halfyear = pd.read_csv("data/benchmark_performance/FUEVN100_HALFYEAR.csv")
vni_halfyear = pd.read_csv("data/benchmark_performance/VNI_HALFYEAR.csv")

fuemav30_fullyear = pd.read_csv("data/benchmark_performance/FUEMAV30_FULLYEAR.csv")
fuessv30_fullyear = pd.read_csv("data/benchmark_performance/FUESSV30_FULLYEAR.csv")
fuessv50_fullyear = pd.read_csv("data/benchmark_performance/FUESSV50_FULLYEAR.csv")
fuevfvnd_fullyear = pd.read_csv("data/benchmark_performance/FUEVFVND_FULLYEAR.csv")
fuevn100_fullyear = pd.read_csv("data/benchmark_performance/FUEVN100_FULLYEAR.csv")
vni_fullyear = pd.read_csv("data/benchmark_performance/VNI_FULLYEAR.csv")

lst_benchmark_halfyear = [
    fuessv30_halfyear, fuessv30_halfyear,
    fuessv50_halfyear, fuevfvnd_halfyear,
    fuevn100_halfyear, vni_halfyear
]
lst_benchmark_fullyear = [
    fuessv30_fullyear, fuessv30_fullyear,
    fuessv50_fullyear, fuevfvnd_fullyear,
    fuevn100_fullyear, vni_fullyear
]

dic_performance_halfyear = {
    "name": [],
    "sharpe_ratio": [],
    "sortino_ratio": [],
    "max_drawdown": [],
    "info_ratio": [],
    "accum_pv": []
}
dic_performance_fullyear = {
    "name": [],
    "sharpe_ratio": [],
    "sortino_ratio": [],
    "max_drawdown": [],
    "info_ratio": [],
    "accum_pv": []
}

for i in range(7):
    if i<6:
        # Halfyear
        measures_halfyear = calc_measures(lst_benchmark_halfyear[i], lst_benchmark_halfyear[0], 0)
        dic_performance_halfyear["name"].append(lst_etf_name[i])
        # Fullyear
        measures_fullyear = calc_measures(lst_benchmark_fullyear[i], lst_benchmark_fullyear[0], 0)
        dic_performance_fullyear["name"].append(lst_etf_name[i])
    else:
        # Halfyear
        df_target = df_performance_halfyear
        measures_halfyear = calc_measures(df_target, lst_benchmark_halfyear[0], 0)
        dic_performance_halfyear["name"].append(f"My Agent")
        # Fullyear
        df_target = df_performance_fullyear
        measures_fullyear = calc_measures(df_target, lst_benchmark_fullyear[0], 0)
        dic_performance_fullyear["name"].append(f"My Agent")
    
    # Halfyear
    dic_performance_halfyear["sharpe_ratio"].append(measures_halfyear[0])
    dic_performance_halfyear["sortino_ratio"].append(measures_halfyear[1])
    dic_performance_halfyear["max_drawdown"].append(measures_halfyear[2])
    dic_performance_halfyear["info_ratio"].append(measures_halfyear[3])
    dic_performance_halfyear["accum_pv"].append(measures_halfyear[4])
    # Fullyear
    dic_performance_fullyear["sharpe_ratio"].append(measures_fullyear[0])
    dic_performance_fullyear["sortino_ratio"].append(measures_fullyear[1])
    dic_performance_fullyear["max_drawdown"].append(measures_fullyear[2])
    dic_performance_fullyear["info_ratio"].append(measures_fullyear[3])
    dic_performance_fullyear["accum_pv"].append(measures_fullyear[4])

df_half = pd.DataFrame(dic_performance_halfyear)
print("Halfyear Relative Performance")
print(df_half)

df_full = pd.DataFrame(dic_performance_fullyear)
print("Fullyear Relative Performance")
print(df_full)