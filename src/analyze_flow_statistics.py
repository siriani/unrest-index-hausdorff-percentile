#!/usr/bin/env python3
"""analyze_flow_statistics.py -- R3 (four optical-flow statistics): per recording, Spearman correlation of the
classic Unrest Index and of tracker-derived travel distance with each of the four
per-step optical-flow statistics (mean, variance, skewness, kurtosis).

Inputs: data/timeseries/<video>_sweep_{mask_v3,bbox}/unrest_timeseries.csv[.gz]
Outputs: <out-dir>/flow_statistics.{csv,md}
Point estimates only (exploratory, as stated in Section 2.6)."""
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

import argparse

BASE = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data-dir", type=Path, default=BASE / "data" / "timeseries")
ap.add_argument("--out-dir", type=Path, default=BASE / "results")
args = ap.parse_args()
RES, OUT = args.data_dir, args.out_dir
VIDEOS = [
    "DVR-Ch1-CAM2-2026-06-18", "DVR-Ch2-CAM1-2026-06-18", "DVR-Ch3-CAM3-2026-06-18",
    "RecordPlan_20260224100150_1007_2", "RecordPlan_20260224140202_1007_3",
    "RecordPlan_20260409084432_1007_3", "RecordPlan_20260423103037_1001_4",
    "SimPlay-LCT2-CAM2-2026-03-16",
]
FLOW = ["optical_flow_mean", "optical_flow_var", "optical_flow_skew", "optical_flow_kurt"]
SERIES = {"unrest_classic": "Unrest Index (classic)", "flock_walking_distance_step_px": "Travel distance"}

rows = []
for v in VIDEOS:
    for src in ("mask_v3", "bbox"):
        fp = RES / f"{v}_sweep_{src}" / "unrest_timeseries.csv"
        df = pd.read_csv(fp if fp.exists() else fp.with_suffix(".csv.gz"))
        df = df[df["step_idx"] > 0]
        for col, label in SERIES.items():
            for f in FLOW:
                d = df[[col, f]].dropna()
                rho = spearmanr(d[col], d[f]).statistic
                rows.append(dict(video=v, centroid=src, series=label, flow_stat=f.replace("optical_flow_", ""),
                                 n_steps=len(d), spearman_rho=round(float(rho), 3)))

t = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
t.to_csv(OUT / "flow_statistics.csv", index=False)
wide = t.pivot_table(index=["centroid", "series", "video"], columns="flow_stat", values="spearman_rho")[
    ["mean", "var", "skew", "kurt"]]
summ = t.groupby(["centroid", "series", "flow_stat"])["spearman_rho"].agg(["median", "min", "max"]).round(3)
with open(OUT / "flow_statistics.md", "w") as fh:
    fh.write("# Spearman rho with the four optical-flow statistics (per recording)\n\n")
    fh.write(wide.to_markdown() + "\n\n## Across recordings (median, range)\n\n" + summ.to_markdown() + "\n")
print(summ.to_string())
