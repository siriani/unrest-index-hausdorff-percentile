#!/usr/bin/env python3
"""
analyze_paired_centroids.py -- controlled centroid comparison
(Animals-4688242: R1 comments 3, 7, 8, 11-13, 16; R3 comments 1, 2, 3, 11).

Input per video (written by src/dump_paired_instances.py; archived on Zenodo as outputs.zip):
  <paired-dir>/<video>/{instances.csv.gz, steps.csv, run_config.json}
and, for the model decomposition, the detector-model run committed in this repository
  <det-dir>/<video>_sweep_bbox/unrest_timeseries.csv[.gz]

ONE segmentation pass (seg_v3 + ByteTrack, exactly the call of the submitted
manuscript) gives, for every tracked instance, three positions:
  kf    centre of the box returned by model.track() = ByteTrack Kalman-filter
        posterior (this is what "box centre" meant in the submitted manuscript,
        for both det_v5 and seg_v3 runs)
  raw   centre of the raw detector box of the same detection
  mask  area centroid of the same detection's mask polygon (cv2.moments)
Frames, instances, confidence threshold and track IDs are identical across the
three read-outs.

Series rebuilt with the semantics of compute_unrest_index.py:
  UI_<src>_<k>  Unrest Index, symmetric Hausdorff (k = classic max or percentile),
                defined when both consecutive steps have >= 1 instance; instances
                returned without a track ID (track_id = -1) enter the UI only.
  WD_<src>      travel distance per step: sum over IDs present at s and s-1.
  flow          mean Farneback block magnitude (detector-independent; verified
                bit-identical to the earlier runs).

Contrasts (Spearman rho differences, paired moving-block bootstrap: one index
set per replicate applied to every series, ranks recomputed per replicate):
  centroid effect   mask - raw   (vs flow, and "matched": each UI vs its own WD)
  smoothing effect  kf   - raw   (same references)
  model effect      UI_kf[seg_v3] - UI_kf[det_v5]  vs flow only (the only
                    reference common to both runs). With centroid(mask-kf) the
                    two terms telescope to the submitted-manuscript contrast;
                    the decomposition is path-dependent (no det_v5 mask arm).
  shared-coordinate share  rho(UI_src, WD_src) - rho(UI_src, WD_other):
                    how much of the UI~WD agreement comes from sharing the same
                    position estimates (still shares detections, IDs, frames).
  percentile        p95 - classic, and the k sweep (descriptive).

Bootstrap: Kunsch (1989), block = round(L * f_eff) steps, L in {90, 10} s
(90 s primary), blocks drawn from all n-b+1 starts without wrap, concatenated
and truncated to n; nboot replicates; numpy default_rng(seed); percentile
95 % intervals. p_boot = CI-inversion two-sided bootstrap p (reported as
< 1/nboot when 0). Holm correction over the pre-specified primary family per
video. Intervals describe within-recording uncertainty only; the eight
recordings are not independent flocks.

Output: <out-dir>/paired_centroids.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "data" / "timeseries"      # detector-model runs (<video>_sweep_bbox)
PAIRED = BASE / "data" / "paired"        # unzip outputs.zip (Zenodo) here

VIDEOS = [  # (id, label, frames in the model-development pool?)
    ("DVR-Ch1-CAM2-2026-06-18", "DVR-Ch1 (principal)", True),
    ("RecordPlan_20260423103037_1001_4", "RecordPlan-04/23", True),
    ("RecordPlan_20260224100150_1007_2", "RecordPlan-02/24a", True),
    ("RecordPlan_20260224140202_1007_3", "RecordPlan-02/24b", True),
    ("RecordPlan_20260409084432_1007_3", "RecordPlan-04/09", False),
    ("SimPlay-LCT2-CAM2-2026-03-16", "SimPlay-CAM2", False),
    ("DVR-Ch2-CAM1-2026-06-18", "DVR-Ch2", False),
    ("DVR-Ch3-CAM3-2026-06-18", "DVR-Ch3", False),
]
KS = [None, 30, 40, 50, 60, 70, 80, 90, 95]  # None = classic (max)
SRCS = ("kf", "raw", "mask")
PRIMARY = [  # pre-specified primary family (Holm within video)
    "centroid(mask-raw)|classic|flow",
    "centroid(mask-raw)|classic|matched",
    "centroid(mask-raw)|p95|flow",
    "centroid(mask-raw)|p95|matched",
    "p95-classic|mask|flow",
    "p95-classic|mask|matched",
]


def kname(k):
    return "classic" if k is None else f"p{k}"


def hausdorff_multi(a, b):
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    ab, ba = d.min(axis=1), d.min(axis=0)
    return {k: (max(ab.max(), ba.max()) if k is None else max(np.percentile(ab, k), np.percentile(ba, k)))
            for k in KS}


def build_series(inst: pd.DataFrame, n_steps: int) -> dict[str, np.ndarray]:
    inst = inst.copy()
    inst["kf_cx"], inst["kf_cy"] = (inst.x1 + inst.x2) / 2, (inst.y1 + inst.y2) / 2
    inst["raw_cx"], inst["raw_cy"] = (inst.raw_x1 + inst.raw_x2) / 2, (inst.raw_y1 + inst.raw_y2) / 2
    groups = {s: g for s, g in inst.groupby("step_idx")}
    S = {f"UI_{src}_{kname(k)}": np.full(n_steps, np.nan) for src in SRCS for k in KS}
    for src in SRCS:
        S[f"WD_{src}"] = np.zeros(n_steps)
    S["n"] = np.zeros(n_steps)
    S["off_raw_mask"] = np.full(n_steps, np.nan)
    S["off_kf_raw"] = np.full(n_steps, np.nan)
    prev = None
    for s in range(n_steps):
        g = groups.get(s)
        if g is not None:
            S["n"][s] = len(g)
            S["off_raw_mask"][s] = float(np.median(np.hypot(g.raw_cx - g.mask_cx, g.raw_cy - g.mask_cy)))
            S["off_kf_raw"][s] = float(np.median(np.hypot(g.kf_cx - g.raw_cx, g.kf_cy - g.raw_cy)))
        if prev is not None and g is not None:
            for src in SRCS:
                a = g[[f"{src}_cx", f"{src}_cy"]].to_numpy()
                b = prev[[f"{src}_cx", f"{src}_cy"]].to_numpy()
                for k, v in hausdorff_multi(a, b).items():
                    S[f"UI_{src}_{kname(k)}"][s] = v
            gt, pt = g[g.track_id >= 0], prev[prev.track_id >= 0]
            m = gt.merge(pt, on="track_id", suffixes=("", "_p"))
            for src in SRCS:
                S[f"WD_{src}"][s] = np.hypot(m[f"{src}_cx"] - m[f"{src}_cx_p"], m[f"{src}_cy"] - m[f"{src}_cy_p"]).sum()
        prev = g
    return S


def sp(ra, rb):
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def paired_bootstrap(V, pairs, diffs, block, nboot, rng):
    n = len(next(iter(V.values())))
    names = sorted({x for p in pairs.values() for x in p})
    R0 = {x: rankdata(V[x]) for x in names}
    point = {p: sp(R0[a], R0[b]) for p, (a, b) in pairs.items()}
    boot = {p: np.empty(nboot) for p in pairs}
    n_blocks = int(np.ceil(n / block))
    for i in range(nboot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        R = {x: rankdata(V[x][idx]) for x in names}
        for p, (a, b) in pairs.items():
            boot[p][i] = sp(R[a], R[b])
    out = {"corr": {}, "diff": {}}
    for p in pairs:
        out["corr"][p] = {"rho": point[p], "ci": [float(np.percentile(boot[p], 2.5)), float(np.percentile(boot[p], 97.5))]}
    for d, (p1, p2) in diffs.items():
        bd = boot[p1] - boot[p2]
        pb = float(min(1.0, 2 * min((bd <= 0).mean(), (bd >= 0).mean())))
        out["diff"][d] = {"delta": point[p1] - point[p2],
                          "ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                          "p_boot": max(pb, 1.0 / nboot), "p_boot_is_bound": pb < 1.0 / nboot}
    prim = {d: out["diff"][d]["p_boot"] for d in PRIMARY if d in out["diff"]}
    for d, ph in holm(prim).items():
        out["diff"][d]["p_holm_primary"] = ph
    return out


def analyse_video(vid, nboot, block_secs, seed):
    d = PAIRED / vid
    cfg = json.loads((d / "run_config.json").read_text())
    steps = pd.read_csv(d / "steps.csv")
    inst = pd.read_csv(d / "instances.csv.gz")
    n_bad_mask = int((inst.mask_ok <= 0).sum())
    n_no_raw = int(inst.raw_x1.isna().sum())
    inst = inst[(inst.mask_ok > 0) & inst.raw_x1.notna()]
    assert not inst.duplicated(["step_idx", "track_id"]).loc[inst.track_id >= 0].any()
    n_steps = len(steps)
    S = build_series(inst, n_steps)
    S["flow"] = steps.optical_flow_mean.to_numpy(float)

    det_csv = RES / f"{vid}_sweep_bbox" / "unrest_timeseries.csv"
    if not det_csv.exists():
        det_csv = det_csv.with_suffix(".csv.gz")
    have_det = det_csv.exists()
    flow_diff = None
    if have_det:
        det = pd.read_csv(det_csv).set_index("step_idx").reindex(range(n_steps))
        nf_ok = np.array_equal(det["native_frame_idx"].to_numpy(float), steps.native_frame_idx.to_numpy(float))
        flow_diff = float(np.nanmax(np.abs(det["optical_flow_mean"].to_numpy(float) - S["flow"])))
        if not nf_ok or flow_diff > 1e-6:
            raise SystemExit(f"{vid}: det_v5 run not frame-aligned (native idx equal={nf_ok}, flow diff={flow_diff})")
        for k in ("classic", "p95"):
            S[f"UI_det_{k}"] = det[f"unrest_{k}"].to_numpy(float)
        S["WD_det"] = det["flock_walking_distance_step_px"].to_numpy(float)

    keys = [k for k in S if k.startswith(("UI_", "WD_")) or k == "flow"]
    valid = np.ones(n_steps, bool)
    for k in keys:
        valid &= np.isfinite(S[k])
    V = {k: S[k][valid] for k in keys}
    t = steps.timestamp_sec.to_numpy(float)
    f_eff = (n_steps - 1) / (t[-1] - t[0])

    pairs, diffs = {}, {}
    for k in KS:
        kn = kname(k)
        for src in SRCS:
            for ref in ("flow", *(f"WD_{s}" for s in SRCS)):
                pairs[f"UI_{src}_{kn}|{ref}"] = (f"UI_{src}_{kn}", ref)
    for kn in [kname(k) for k in KS]:
        for a, b, lab in (("mask", "raw", "centroid(mask-raw)"), ("kf", "raw", "smoothing(kf-raw)"),
                          ("mask", "kf", "centroid(mask-kf)")):
            diffs[f"{lab}|{kn}|flow"] = (f"UI_{a}_{kn}|flow", f"UI_{b}_{kn}|flow")
            diffs[f"{lab}|{kn}|matched"] = (f"UI_{a}_{kn}|WD_{a}", f"UI_{b}_{kn}|WD_{b}")
    for kn in ("classic", "p95"):
        for a, b in (("mask", "raw"), ("raw", "mask"), ("kf", "raw"), ("raw", "kf")):
            diffs[f"shared({a} vs WD_{b})|{kn}"] = (f"UI_{a}_{kn}|WD_{a}", f"UI_{a}_{kn}|WD_{b}")
    for src in SRCS:
        diffs[f"p95-classic|{src}|flow"] = (f"UI_{src}_p95|flow", f"UI_{src}_classic|flow")
        diffs[f"p95-classic|{src}|matched"] = (f"UI_{src}_p95|WD_{src}", f"UI_{src}_classic|WD_{src}")
    for a, b in (("mask", "flow"), ("raw", "flow"), ("kf", "flow"), ("mask", "raw"), ("kf", "raw")):
        pairs[f"WD_{a}|{'flow' if b == 'flow' else 'WD_' + b}"] = (f"WD_{a}", "flow" if b == "flow" else f"WD_{b}")
    if have_det:
        for kn in ("classic", "p95"):
            pairs[f"UI_det_{kn}|flow"] = (f"UI_det_{kn}", "flow")
            pairs[f"UI_det_{kn}|WD_det"] = (f"UI_det_{kn}", "WD_det")
            diffs[f"model(seg_kf-det_kf)|{kn}|flow"] = (f"UI_kf_{kn}|flow", f"UI_det_{kn}|flow")
            diffs[f"submitted(seg_mask-det_kf)|{kn}|flow"] = (f"UI_mask_{kn}|flow", f"UI_det_{kn}|flow")
            diffs[f"submitted(seg_mask-det_kf)|{kn}|own_WD"] = (f"UI_mask_{kn}|WD_mask", f"UI_det_{kn}|WD_det")

    lab = dict((v, l) for v, l, _ in VIDEOS)
    pool = dict((v, p) for v, _, p in VIDEOS)
    res = {
        "label": lab[vid], "in_dev_pool": pool[vid],
        "steps_total": n_steps, "steps_valid": int(valid.sum()), "f_eff_hz": f_eff,
        "n_track_ids": int(inst.loc[inst.track_id >= 0, "track_id"].nunique()),
        "n_untracked_instances": int((inst.track_id < 0).sum()),
        "n_instances_dropped_bad_mask": n_bad_mask, "n_instances_dropped_no_raw_box": n_no_raw,
        "raw_box_ambiguous_score_matches": cfg.get("raw_box_ambiguous_score_matches"),
        "raw_box_idx_crosscheck_mismatches": cfg.get("raw_box_idx_crosscheck_mismatches"),
        "n_birds_median": float(np.median(S["n"])),
        "offset_raw_mask_px_median": float(np.nanmedian(S["off_raw_mask"])),
        "offset_kf_raw_px_median": float(np.nanmedian(S["off_kf_raw"])),
        "offset_kf_raw_px_p95": float(np.nanpercentile(S["off_kf_raw"], 95)),
        "flow_max_abs_diff_vs_det_run": flow_diff,
        "bootstrap": {},
    }
    for i, L in enumerate(block_secs):
        block = max(4, int(round(L * f_eff)))
        rng = np.random.default_rng(seed + i)
        print(f"  {vid}: block {L:g}s = {block} steps", flush=True)
        res["bootstrap"][f"{L:g}s"] = {"block_steps": block, "seed": seed + i,
                                        **paired_bootstrap(V, pairs, diffs, block, nboot, rng)}
    return res


def across_videos(all_res, L):
    keys = set()
    for r in all_res["videos"].values():
        keys |= set(r["bootstrap"][L]["diff"])
    out = {}
    for k in sorted(keys):
        ds = [r["bootstrap"][L]["diff"][k] for r in all_res["videos"].values() if k in r["bootstrap"][L]["diff"]]
        deltas = np.array([d["delta"] for d in ds])
        out[k] = {"n_videos": len(ds), "mean": float(deltas.mean()), "min": float(deltas.min()),
                  "max": float(deltas.max()), "n_positive": int((deltas > 0).sum()),
                  "n_ci_above_0": int(sum(d["ci"][0] > 0 for d in ds)),
                  "n_ci_below_0": int(sum(d["ci"][1] < 0 for d in ds))}
    return out


def f3(x):
    return f"{x:+.3f}"


def cell_d(b, k):
    if k not in b:
        return "--"
    d = b[k]
    star = ""
    if "p_holm_primary" in d and d["p_holm_primary"] < 0.05:
        star = "*"
    return f"{f3(d['delta'])} [{f3(d['ci'][0])}, {f3(d['ci'][1])}]{star}"


def cell_c(c, k):
    return f"{c[k]['rho']:.3f} [{c[k]['ci'][0]:.3f}, {c[k]['ci'][1]:.3f}]" if k in c else "--"


def emit_md(all_res, out):
    L = all_res["meta"]["primary_block"]
    v = all_res["videos"]
    o = ["# Controlled centroid comparison (same seg_v3 instances, IDs and frames)", "",
         f"Spearman rho. Paired moving-block bootstrap, primary block {L} "
         f"({all_res['meta']['nboot']} replicates, seed {all_res['meta']['seed']}), percentile 95% CI. "
         "`*` = primary contrast significant after Holm (within video). Within-recording uncertainty only.", "",
         "Read-outs: **kf** = centre of the box returned by model.track() (ByteTrack Kalman posterior; "
         "the 'box centre' of the submitted manuscript); **raw** = raw detector box centre; "
         "**mask** = mask area centroid. **det** = earlier det_v5 run (kf box).", "",
         "## 0. Data checks", "",
         "| Video | dev pool | steps valid | f_eff Hz | track IDs | untracked inst. | med. birds | dist raw–mask px | dist kf–raw px (med / p95) | ambiguous / idx mismatch | dropped: bad mask / no raw box |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in v.values():
        o.append(f"| {r['label']} | {'yes' if r['in_dev_pool'] else 'no'} | {r['steps_valid']} | {r['f_eff_hz']:.2f} | "
                 f"{r['n_track_ids']} | {r['n_untracked_instances']} | {r['n_birds_median']:.0f} | "
                 f"{r['offset_raw_mask_px_median']:.2f} | {r['offset_kf_raw_px_median']:.2f} / {r['offset_kf_raw_px_p95']:.2f} | "
                 f"{r['raw_box_ambiguous_score_matches']} / {r['raw_box_idx_crosscheck_mismatches']} | "
                 f"{r['n_instances_dropped_bad_mask']} / {r['n_instances_dropped_no_raw_box']} |")
    for kn in ("classic", "p95"):
        o += ["", f"## 1. Correlations, {kn} Hausdorff (rho [95% CI])", "",
              "| Video | UI_kf~WD_kf | UI_raw~WD_raw | UI_mask~WD_mask | UI_det~WD_det | UI_kf~flow | UI_raw~flow | UI_mask~flow | UI_det~flow |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in v.values():
            c = r["bootstrap"][L]["corr"]
            o.append(f"| {r['label']} | {cell_c(c, f'UI_kf_{kn}|WD_kf')} | {cell_c(c, f'UI_raw_{kn}|WD_raw')} | "
                     f"{cell_c(c, f'UI_mask_{kn}|WD_mask')} | {cell_c(c, f'UI_det_{kn}|WD_det')} | "
                     f"{cell_c(c, f'UI_kf_{kn}|flow')} | {cell_c(c, f'UI_raw_{kn}|flow')} | "
                     f"{cell_c(c, f'UI_mask_{kn}|flow')} | {cell_c(c, f'UI_det_{kn}|flow')} |")
    for kn in ("classic", "p95"):
        o += ["", f"## 2. Effects, {kn} Hausdorff (delta rho [95% CI])", "",
              "| Video | centroid mask-raw vs flow | centroid mask-raw matched | smoothing kf-raw vs flow | smoothing kf-raw matched | model seg-det (kf) vs flow | submitted contrast vs flow | submitted contrast own WD |",
              "|---|---|---|---|---|---|---|---|"]
        for r in v.values():
            b = r["bootstrap"][L]["diff"]
            o.append(f"| {r['label']} | {cell_d(b, f'centroid(mask-raw)|{kn}|flow')} | {cell_d(b, f'centroid(mask-raw)|{kn}|matched')} | "
                     f"{cell_d(b, f'smoothing(kf-raw)|{kn}|flow')} | {cell_d(b, f'smoothing(kf-raw)|{kn}|matched')} | "
                     f"{cell_d(b, f'model(seg_kf-det_kf)|{kn}|flow')} | {cell_d(b, f'submitted(seg_mask-det_kf)|{kn}|flow')} | "
                     f"{cell_d(b, f'submitted(seg_mask-det_kf)|{kn}|own_WD')} |")
    o += ["", "## 3. Shared-coordinate share of UI~WD agreement, classic (rho(UI_a,WD_a) - rho(UI_a,WD_b))", "",
          "| Video | mask: WD_mask vs WD_raw | raw: WD_raw vs WD_mask | kf: WD_kf vs WD_raw | raw: WD_raw vs WD_kf |",
          "|---|---|---|---|---|"]
    for r in v.values():
        b = r["bootstrap"][L]["diff"]
        o.append(f"| {r['label']} | {cell_d(b, 'shared(mask vs WD_raw)|classic')} | {cell_d(b, 'shared(raw vs WD_mask)|classic')} | "
                 f"{cell_d(b, 'shared(kf vs WD_raw)|classic')} | {cell_d(b, 'shared(raw vs WD_kf)|classic')} |")
    o += ["", "## 4. p95 - classic (delta rho [95% CI])", "",
          "| Video | " + " | ".join(f"{s} {ref}" for s in SRCS for ref in ("vs flow", "matched")) + " |", "|---|" + "---|" * (2 * len(SRCS))]
    for r in v.values():
        b = r["bootstrap"][L]["diff"]
        o.append("| " + r["label"] + " | " + " | ".join(cell_d(b, f"p95-classic|{s}|{ref}") for s in SRCS for ref in ("flow", "matched")) + " |")
    o += ["", f"## 5. Across the {len(v)} videos (block {L})", "",
          "| Contrast | mean | min | max | n>0 | CI>0 | CI<0 |", "|---|---|---|---|---|---|---|"]
    for k, s in all_res["across_videos"][L].items():
        if k.split("|")[1:2] and k.split("|")[1] not in ("classic", "p95") and not k.startswith(("p95", "shared")):
            continue
        o.append(f"| {k.replace('|', ' · ')} | {f3(s['mean'])} | {f3(s['min'])} | {f3(s['max'])} | {s['n_positive']}/{s['n_videos']} | "
                 f"{s['n_ci_above_0']} | {s['n_ci_below_0']} |")
    o += ["", "## 6. Block-length sensitivity (classic, centroid mask-raw)", "",
          "| Video | " + " | ".join(f"vs flow {b}" for b in all_res["meta"]["block_keys"]) + " | "
          + " | ".join(f"matched {b}" for b in all_res["meta"]["block_keys"]) + " |",
          "|---|" + "---|" * (2 * len(all_res["meta"]["block_keys"]))]
    for r in v.values():
        cells = [cell_d(r["bootstrap"][bk]["diff"], "centroid(mask-raw)|classic|flow") for bk in all_res["meta"]["block_keys"]]
        cells += [cell_d(r["bootstrap"][bk]["diff"], "centroid(mask-raw)|classic|matched") for bk in all_res["meta"]["block_keys"]]
        o.append(f"| {r['label']} | " + " | ".join(cells) + " |")
    out.write_text("\n".join(o) + "\n", "utf-8")


def main():
    global PAIRED, RES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--blocks", default="90,10", help="block lengths in s; the first is primary")
    ap.add_argument("--videos", default=None)
    ap.add_argument("--paired-dir", type=Path, default=PAIRED, help="per-video outputs of dump_paired_instances.py")
    ap.add_argument("--det-dir", type=Path, default=RES, help="folder holding <video>_sweep_bbox/ detector runs")
    ap.add_argument("--out-dir", type=Path, default=BASE / "results")
    args = ap.parse_args()
    PAIRED, RES = args.paired_dir, args.det_dir
    block_secs = [float(x) for x in args.blocks.split(",")]
    sel = args.videos.split(",") if args.videos else [v for v, _, _ in VIDEOS]
    bkeys = [f"{b:g}s" for b in block_secs]
    all_res = {"meta": {"nboot": args.nboot, "seed": args.seed, "block_keys": bkeys, "primary_block": bkeys[0],
                        "ks": [kname(k) for k in KS], "primary_family": PRIMARY,
                        "notes": ["kf, raw and mask read-outs come from the same seg_v3 tracked instances.",
                                  "k sweep beyond p95 vs classic is descriptive (no multiplicity correction).",
                                  "CIs are within-recording; videos are not independent flocks.",
                                  "Decomposition model+centroid telescopes but is path-dependent (no det_v5 mask arm)."]},
               "videos": {}}
    for vid in sel:
        if not (PAIRED / vid / "run_config.json").exists():
            print("missing", vid)
            continue
        print("analysing", vid, flush=True)
        all_res["videos"][vid] = analyse_video(vid, args.nboot, block_secs, args.seed)
    all_res["across_videos"] = {bk: across_videos(all_res, bk) for bk in bkeys}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "paired_centroids.json").write_text(json.dumps(all_res, indent=2, ensure_ascii=False), "utf-8")
    emit_md(all_res, args.out_dir / "paired_centroids.md")
    print((args.out_dir / "paired_centroids.md").read_text())


if __name__ == "__main__":
    main()
