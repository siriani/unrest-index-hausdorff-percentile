#!/usr/bin/env python3
"""
analyze_roi_sensitivity.py -- region-of-interest sensitivity analysis
(Animals-4688242: R1 comment 10, R3 comment 7).

For each recording a fixed polygon covering the pen floor (litter) is applied
offline to the outputs of dump_paired_instances.py:

  * centroid sets: an instance is kept when its MASK area centroid lies inside
    the polygon or within --tolerance-px of its edge (birds against the wall
    have their projected centroid slightly outside the litter). The same
    instances are then used for all three read-outs (kf, raw, mask), so the
    read-out comparison stays paired.
  * optical flow: mean of the per-block flow magnitudes (flow_grid.npz,
    16-px blocks of the half-resolution frame) whose block centre lies inside
    the polygon. The full-frame counterpart is the mean over all blocks.

The Unrest Index and the tracker-derived travel distance are rebuilt with the
same code as analyze_paired_centroids.py. Agreement with and without the ROI
is compared with the same paired moving-block bootstrap (one set of block
indices per replicate for every series).

Inputs : <paired-dir>/<video>/{instances.csv.gz, steps.csv, flow_grid.npz}
         --roi-json  {video: [[x, y], ...]} in native pixels
Output : <out-dir>/roi_sensitivity.{json,md}
"""
from __future__ import annotations

import argparse
import warnings
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_paired_centroids as apc  # noqa: E402

warnings.filterwarnings("ignore", message="Mean of empty slice")

SRCS = ("kf", "raw", "mask")
KNS = ("classic", "p95")


def flow_in_roi(grid_npz: Path, poly: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    z = np.load(grid_npz)
    g = z["grid"].astype(np.float32)               # (n_steps, rows, cols)
    cell = float(z["block_px_downscaled"]) / float(z["flow_scale"])   # block size in native px
    rows, cols = g.shape[1:]
    cy = (np.arange(rows) + 0.5) * cell
    cx = (np.arange(cols) + 0.5) * cell
    inside = np.array([[cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x in cx] for y in cy])
    with np.errstate(invalid="ignore"):
        full = np.nanmean(g.reshape(len(g), -1), axis=1)
        roi = np.nanmean(g[:, inside], axis=1)
    return full, roi, int(inside.sum()), int(inside.size)


def analyse(vid: str, poly: np.ndarray, tol: float, nboot: int, block_sec: float, seed: int, paired_dir: Path):
    d = paired_dir / vid
    steps = pd.read_csv(d / "steps.csv")
    inst = pd.read_csv(d / "instances.csv.gz")
    inst = inst[(inst.mask_ok > 0) & inst.raw_x1.notna()].copy()
    n_steps = len(steps)
    dist = np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), True)
                     for x, y in zip(inst.mask_cx, inst.mask_cy)])
    keep = dist >= -tol
    roi_inst = inst[keep]

    S_full = apc.build_series(inst, n_steps)
    S_roi = apc.build_series(roi_inst, n_steps)
    flow_grid_full, flow_roi, n_blk_in, n_blk = flow_in_roi(d / "flow_grid.npz", poly)
    flow_ref = steps.optical_flow_mean.to_numpy(float)

    V = {"flow": flow_ref, "flowG_full": flow_grid_full, "flow_roi": flow_roi}
    for src in SRCS:
        for kn in KNS:
            V[f"UI_{src}_{kn}"] = S_full[f"UI_{src}_{kn}"]
            V[f"UIroi_{src}_{kn}"] = S_roi[f"UI_{src}_{kn}"]
        V[f"WD_{src}"] = S_full[f"WD_{src}"]
        V[f"WDroi_{src}"] = S_roi[f"WD_{src}"]
    valid = np.ones(n_steps, bool)
    for v in V.values():
        valid &= np.isfinite(v)
    V = {k: v[valid] for k, v in V.items()}

    pairs, diffs = {}, {}
    pairs["flow|flowG_full"] = ("flow", "flowG_full")
    pairs["flow|flow_roi"] = ("flow", "flow_roi")
    for src in SRCS:
        for kn in KNS:
            pairs[f"UI_{src}_{kn}|flow"] = (f"UI_{src}_{kn}", "flow")
            pairs[f"UIroi_{src}_{kn}|flow_roi"] = (f"UIroi_{src}_{kn}", "flow_roi")
            pairs[f"UIroi_{src}_{kn}|flow"] = (f"UIroi_{src}_{kn}", "flow")
            pairs[f"UI_{src}_{kn}|WD_{src}"] = (f"UI_{src}_{kn}", f"WD_{src}")
            pairs[f"UIroi_{src}_{kn}|WDroi_{src}"] = (f"UIroi_{src}_{kn}", f"WDroi_{src}")
            diffs[f"roi-full|{src}|{kn}|flow"] = (f"UIroi_{src}_{kn}|flow_roi", f"UI_{src}_{kn}|flow")
            diffs[f"roi_points_only|{src}|{kn}|flow"] = (f"UIroi_{src}_{kn}|flow", f"UI_{src}_{kn}|flow")
            diffs[f"roi-full|{src}|{kn}|matched"] = (f"UIroi_{src}_{kn}|WDroi_{src}", f"UI_{src}_{kn}|WD_{src}")
        pairs[f"WD_{src}|flow"] = (f"WD_{src}", "flow")
        pairs[f"WDroi_{src}|flow_roi"] = (f"WDroi_{src}", "flow_roi")
        diffs[f"roi-full|WD_{src}|flow"] = (f"WDroi_{src}|flow_roi", f"WD_{src}|flow")
    for kn in KNS:
        diffs[f"centroid_roi(mask-raw)|{kn}|flow"] = (f"UIroi_mask_{kn}|flow_roi", f"UIroi_raw_{kn}|flow_roi")
        diffs[f"centroid_full(mask-raw)|{kn}|flow"] = (f"UI_mask_{kn}|flow", f"UI_raw_{kn}|flow")

    t = steps.timestamp_sec.to_numpy(float)
    f_eff = (n_steps - 1) / (t[-1] - t[0])
    block = max(4, int(round(block_sec * f_eff)))
    rng = np.random.default_rng(seed)
    apc.PRIMARY = []  # no Holm family here: descriptive sensitivity analysis
    boot = apc.paired_bootstrap(V, pairs, diffs, block, nboot, rng)
    n_per_step_full = inst.groupby("step_idx").size().reindex(range(n_steps), fill_value=0)
    n_per_step_roi = roi_inst.groupby("step_idx").size().reindex(range(n_steps), fill_value=0)
    return {
        "polygon_px": poly.reshape(-1, 2).tolist(), "tolerance_px": tol,
        "instances_total": int(len(inst)), "instances_removed": int((~keep).sum()),
        "instances_removed_pct": float(100 * (~keep).mean()),
        "instances_removed_untracked": int(((~keep) & (inst.track_id < 0)).sum()),
        "steps_with_removal": int((n_per_step_full != n_per_step_roi).sum()),
        "median_birds_full": float(n_per_step_full.median()), "median_birds_roi": float(n_per_step_roi.median()),
        "flow_blocks_in_roi": n_blk_in, "flow_blocks_total": n_blk,
        "steps_total": n_steps, "steps_valid": int(valid.sum()), "f_eff_hz": f_eff,
        "block_sec": block_sec, "block_steps": block, "seed": seed, "nboot": nboot,
        **boot,
    }


def emit_md(R: dict, out: Path):
    lab = dict((v, l) for v, l, _ in apc.VIDEOS)
    o = ["# Region-of-interest sensitivity (fixed pen-floor polygon)", "",
         f"Spearman rho; paired moving-block bootstrap, block {R['meta']['block_sec']:g} s, "
         f"{R['meta']['nboot']} replicates, seed {R['meta']['seed']}; percentile 95 % CI. "
         f"Instances kept when the mask centroid is inside the polygon or within {R['meta']['tolerance_px']:g} px of it; "
         "optical flow averaged over 16-px blocks (half-resolution frame) whose centre is inside the polygon. "
         "Within-recording uncertainty only.", "",
         "## 0. What the ROI removes", "",
         "| Video | instances removed (%) | of which untracked | steps affected | median birds full / ROI | flow blocks in ROI | rho(flow, grid mean) | rho(flow, ROI flow) |",
         "|---|---|---|---|---|---|---|---|"]
    for v, r in R["videos"].items():
        c = r["corr"]
        o.append(f"| {lab[v]} | {r['instances_removed']} ({r['instances_removed_pct']:.2f}) | {r['instances_removed_untracked']} | "
                 f"{r['steps_with_removal']} | {r['median_birds_full']:.0f} / {r['median_birds_roi']:.0f} | "
                 f"{r['flow_blocks_in_roi']}/{r['flow_blocks_total']} | {c['flow|flowG_full']['rho']:.3f} | {c['flow|flow_roi']['rho']:.3f} |")
    for kn in KNS:
        o += ["", f"## 1. Unrest Index vs optical flow, {kn} Hausdorff", "",
              "| Video | " + " | ".join(f"{s}: full | {s}: ROI | {s}: delta [CI]" for s in SRCS) + " |",
              "|---|" + "---|" * (3 * len(SRCS))]
        for v, r in R["videos"].items():
            c, dd = r["corr"], r["diff"]
            cells = []
            for s in SRCS:
                cells += [f"{c[f'UI_{s}_{kn}|flow']['rho']:.3f}", f"{c[f'UIroi_{s}_{kn}|flow_roi']['rho']:.3f}",
                          apc.cell_d(dd, f"roi-full|{s}|{kn}|flow").replace("*", "")]
            o.append(f"| {lab[v]} | " + " | ".join(cells) + " |")
    o += ["", "## 2. Unrest Index vs own travel distance (classic)", "",
          "| Video | " + " | ".join(f"{s}: full | {s}: ROI | {s}: delta [CI]" for s in SRCS) + " |",
          "|---|" + "---|" * (3 * len(SRCS))]
    for v, r in R["videos"].items():
        c, dd = r["corr"], r["diff"]
        cells = []
        for s in SRCS:
            cells += [f"{c[f'UI_{s}_classic|WD_{s}']['rho']:.3f}", f"{c[f'UIroi_{s}_classic|WDroi_{s}']['rho']:.3f}",
                      apc.cell_d(dd, f"roi-full|{s}|classic|matched").replace("*", "")]
        o.append(f"| {lab[v]} | " + " | ".join(cells) + " |")
    o += ["", "## 3. Centroid effect (mask - raw) vs optical flow, full frame and within the ROI", "",
          "| Video | classic full | classic ROI | p95 full | p95 ROI |", "|---|---|---|---|---|"]
    for v, r in R["videos"].items():
        dd = r["diff"]
        o.append(f"| {lab[v]} | " + " | ".join(apc.cell_d(dd, f"centroid_{w}(mask-raw)|{kn}|flow").replace("*", "")
                                              for kn in KNS for w in ("full", "roi")) + " |")
    o += ["", "## 4. Across recordings", "", "| Contrast | mean | min | max | CI>0 | CI<0 |", "|---|---|---|---|---|---|"]
    keys = sorted(next(iter(R["videos"].values()))["diff"].keys())
    for k in keys:
        ds = [r["diff"][k] for r in R["videos"].values()]
        x = np.array([d["delta"] for d in ds])
        o.append(f"| {k.replace('|', ' · ')} | {x.mean():+.3f} | {x.min():+.3f} | {x.max():+.3f} | "
                 f"{sum(d['ci'][0] > 0 for d in ds)} | {sum(d['ci'][1] < 0 for d in ds)} |")
    out.write_text("\n".join(o) + "\n", "utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paired-dir", type=Path, default=apc.PAIRED)
    ap.add_argument("--roi-json", type=Path, required=True)
    ap.add_argument("--tolerance-px", type=float, default=15.0)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--block-sec", type=float, default=90.0)
    ap.add_argument("--videos", default=None)
    ap.add_argument("--out-dir", type=Path, default=apc.BASE / "results")
    a = ap.parse_args()
    polys = json.loads(a.roi_json.read_text())
    vids = a.videos.split(",") if a.videos else [v for v, _, _ in apc.VIDEOS]
    R = {"meta": {"nboot": a.nboot, "seed": a.seed, "block_sec": a.block_sec, "tolerance_px": a.tolerance_px,
                  "roi_json": a.roi_json.name}, "videos": {}}
    for v in vids:
        print(f"analysing {v}", flush=True)
        R["videos"][v] = analyse(v, np.array(polys[v], np.float32).reshape(-1, 1, 2), a.tolerance_px,
                                 a.nboot, a.block_sec, a.seed, a.paired_dir)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "roi_sensitivity.json").write_text(json.dumps(R, indent=2), "utf-8")
    emit_md(R, a.out_dir / "roi_sensitivity.md")
    print((a.out_dir / "roi_sensitivity.md").read_text())


if __name__ == "__main__":
    main()
