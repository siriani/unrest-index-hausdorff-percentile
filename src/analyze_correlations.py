#!/usr/bin/env python3
"""
analyze_correlations.py -- step-wise cross-correlation between the three
flock-movement metrics for the main analysed video, in the four reported
configurations (centroid source x Hausdorff variant).

Reads the two percentile-sweep time series of the main video:

    <data-dir>/DVR-Ch1-CAM2-2026-06-18_sweep_bbox/unrest_timeseries.csv
    <data-dir>/DVR-Ch1-CAM2-2026-06-18_sweep_mask_v3/unrest_timeseries.csv

Each CSV holds one inference pass with every percentile column
(unrest_classic, unrest_p30 ... unrest_p95). The four configurations are:

    bbox  + classic  ->  _sweep_bbox     column unrest_classic
    bbox  + p95      ->  _sweep_bbox     column unrest_p95
    mask  + classic  ->  _sweep_mask_v3  column unrest_classic
    mask  + p50      ->  _sweep_mask_v3  column unrest_p50

For each configuration it computes Spearman (rho) and Pearson (r) between the
three metric pairs, with 95% CIs from a moving-block bootstrap (Kuensch 1989),
plus the integrated autocorrelation time of the base series and the
co-monotonicity of the index series against the bbox+classic reference. It
also runs a paired moving-block bootstrap for the mask-vs-bbox rho gain.

Outputs:
    <out-dir>/correlations.json
    <out-dir>/tables/tab_correlacoes.tex   (LaTeX fragment, pt-BR to match the manuscript)

The paper reports Spearman as the primary statistic (the series are
heavy-tailed, so Pearson is dominated by a handful of extreme steps); Pearson
is kept in the table only for comparability with Pereira et al. (2025).

Usage:
    python3 src/analyze_correlations.py [--data-dir data/timeseries] [--out-dir results]
                                        [--nboot 2000] [--block-sec 10] [--seed 0]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scipy import stats

MAIN_VIDEO = "DVR-Ch1-CAM2-2026-06-18"

# key -> (sweep subfolder, index column, table label)
CONFIGS = [
    ("bbox_classic", f"{MAIN_VIDEO}_sweep_bbox",    "unrest_classic", "Caixa + clássico"),
    ("bbox_p95",     f"{MAIN_VIDEO}_sweep_bbox",    "unrest_p95",     "Caixa + p95"),
    ("mask_classic", f"{MAIN_VIDEO}_sweep_mask_v3", "unrest_classic", "Máscara + clássico"),
    ("mask_p50",     f"{MAIN_VIDEO}_sweep_mask_v3", "unrest_p50",     "Máscara + p50"),
]
REF_KEY = "bbox_classic"          # reference for series co-monotonicity
WALK = "flock_walking_distance_step_px"
FLOW = "optical_flow_mean"

PAIRS = [
    ("index", FLOW, "Unrest Index x optical flow"),
    ("index", WALK, "Unrest Index x walking distance"),
    (FLOW,    WALK, "optical flow x walking distance"),
]
PAIR_TEX = {
    "Unrest Index x optical flow":      "$\\II$ $\\times$ \\emph{optical flow}",
    "Unrest Index x walking distance":  "$\\II$ $\\times$ dist.\\ percorrida",
    "optical flow x walking distance":  "\\emph{optical flow} $\\times$ dist.\\ percorrida",
}


def load_csv(path: Path) -> dict[str, np.ndarray]:
    """Read a plain or gzip-compressed CSV of floats into column arrays."""
    if not path.exists() and path.with_suffix(path.suffix + ".gz").exists():
        path = path.with_suffix(path.suffix + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    cols: dict[str, list] = {}
    with opener(path, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
    out: dict[str, np.ndarray] = {}
    for k, vals in cols.items():
        try:
            out[k] = np.array([np.nan if v in ("", "nan", "NaN") else float(v) for v in vals])
        except ValueError:
            out[k] = np.array(vals, dtype=object)
    return out


def _clean(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def point_corr(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a, b = _clean(a, b)
    return {
        "n": int(a.size),
        "pearson": float(stats.pearsonr(a, b)[0]),
        "spearman": float(stats.spearmanr(a, b)[0]),
    }


def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def block_bootstrap_ci(a: np.ndarray, b: np.ndarray, *, block: int, nboot: int,
                       rng: np.random.Generator) -> dict[str, list[float]]:
    """95% CI of Pearson and Spearman via a moving-block bootstrap."""
    a, b = _clean(a, b)
    n = a.size
    if n < block * 3:
        return {"pearson_ci": [np.nan, np.nan], "spearman_ci": [np.nan, np.nan]}
    pear = np.empty(nboot)
    spear = np.empty(nboot)
    for i in range(nboot):
        idx = _block_indices(n, block, rng)
        pear[i] = stats.pearsonr(a[idx], b[idx])[0]
        spear[i] = stats.spearmanr(a[idx], b[idx])[0]
    return {
        "pearson_ci": [float(np.percentile(pear, 2.5)), float(np.percentile(pear, 97.5))],
        "spearman_ci": [float(np.percentile(spear, 2.5)), float(np.percentile(spear, 97.5))],
    }


def paired_delta_rho_ci(a1: np.ndarray, ref1: np.ndarray,
                        a2: np.ndarray, ref2: np.ndarray, *,
                        block: int, nboot: int, rng: np.random.Generator) -> dict:
    """
    Moving-block bootstrap for delta-rho = rho(a2, ref2) - rho(a1, ref1),
    resampling the same block indices for every series (paired). Used for the
    mask-vs-bbox gain in rho(Unrest Index, own walking distance) reported in
    Section 4.3 of the manuscript.
    """
    m = np.isfinite(a1) & np.isfinite(ref1) & np.isfinite(a2) & np.isfinite(ref2)
    a1, ref1, a2, ref2 = a1[m], ref1[m], a2[m], ref2[m]
    n = a1.size
    d0 = float(stats.spearmanr(a2, ref2)[0] - stats.spearmanr(a1, ref1)[0])
    dd = np.empty(nboot)
    for i in range(nboot):
        idx = _block_indices(n, block, rng)
        dd[i] = (stats.spearmanr(a2[idx], ref2[idx])[0]
                 - stats.spearmanr(a1[idx], ref1[idx])[0])
    return {"delta_rho": d0, "block_steps": block,
            "ci": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}


def integrated_autocorr_time(x: np.ndarray) -> dict[str, float]:
    x = x[np.isfinite(x)]
    x = x - x.mean()
    if x.std() == 0:
        return {"lag1": float("nan"), "tau_int": float("nan"), "n_eff": float("nan")}
    n = x.size
    ac = np.correlate(x, x, mode="full")[n - 1:]
    ac = ac / ac[0]
    tau = 1.0
    for k in range(1, len(ac)):
        if k > 6 * tau:
            break
        tau += 2 * ac[k]
    tau = float(max(tau, 1.0))
    return {"lag1": float(ac[1]), "tau_int": tau, "n_eff": float(n / tau)}


def fmt(x: float, d: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{d}f}".replace(".", ",")


def emit_latex(res: dict, out: Path) -> None:
    pp = res["step_wise"]
    labels = res["meta"]["config_labels"]
    L = [
        "% Gerado por src/analyze_correlations.py -- nao editar a mao.",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Correlação passo a passo entre as métricas de movimento do lote, "
        "por configuração (fonte de centroide $\\times$ variante de Hausdorff): "
        "Spearman ($\\rho$) e Pearson ($r$), com IC \\SI{95}{\\percent} por "
        "\\emph{bootstrap} em blocos de "
        f"$\\approx$\\SI{{{int(res['meta']['block_sec'])}}}{{\\second}}.}}",
        "  \\label{tab:corr-stats}",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{3pt}",
        "  \\resizebox{\\linewidth}{!}{%",
        "  \\begin{tabular}{llcc}",
        "    \\toprule",
        "    Configuração & Par de métricas & $\\rho$ (IC 95\\%) & $r$ (IC 95\\%) \\\\",
        "    \\midrule",
    ]
    for key, label in labels.items():
        for _, _, pair in PAIRS:
            d = pp[key][pair]
            rho = f"{fmt(d['spearman'])} ({fmt(d['spearman_ci'][0])};\\,{fmt(d['spearman_ci'][1])})"
            r = f"{fmt(d['pearson'])} ({fmt(d['pearson_ci'][0])};\\,{fmt(d['pearson_ci'][1])})"
            L.append(f"    {label} & {PAIR_TEX[pair]} & {rho} & {r} \\\\")
        L.append("    \\midrule")
    L[-1] = "    \\bottomrule"
    L += ["  \\end{tabular}}"]
    comono = [f"{labels[k]} $\\rho = \\num{{{fmt(v['spearman'])}}}$"
              for k, v in res["co_monotonicity"].items()]
    if comono:
        L.append("  \\par\\medskip\\footnotesize Co-monotonicidade das séries do índice "
                 f"(Spearman contra {labels[REF_KEY]}): " + "; ".join(comono) + ".")
    L += ["\\end{table}", ""]
    for k, v in res["co_monotonicity"].items():
        L.append(f"% Co-monotonicidade {labels[k]} vs {labels[REF_KEY]}: "
                 f"rho = {fmt(v['spearman'])}, r = {fmt(v['pearson'])} (n = {v['n']}).")
    out.write_text("\n".join(L) + "\n", "utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/timeseries"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--block-sec", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # load the two sweep CSVs and build the four config series
    raw: dict[str, dict[str, np.ndarray]] = {}
    series: dict[str, dict[str, np.ndarray]] = {}
    labels: dict[str, str] = {}
    for key, folder, col, label in CONFIGS:
        path = args.data_dir / folder / "unrest_timeseries.csv"
        if folder not in raw:
            raw[folder] = load_csv(path)
        s = raw[folder]
        series[key] = {"index": s[col], WALK: s[WALK], FLOW: s[FLOW],
                       "timestamp_sec": s["timestamp_sec"]}
        labels[key] = label

    ref = series[REF_KEY]
    t = ref["timestamp_sec"][np.isfinite(ref["timestamp_sec"])]
    dur = float(t[-1] - t[0])
    n_steps = int(np.isfinite(ref["index"]).sum())
    fs = n_steps / dur
    block = max(4, int(round(args.block_sec * fs)))

    res: dict = {
        "meta": {
            "video": MAIN_VIDEO, "duration_s": dur, "valid_steps": n_steps,
            "effective_rate_hz": fs, "bootstrap_block_steps": block,
            "block_sec": args.block_sec, "nboot": args.nboot, "seed": args.seed,
            "config_labels": labels,
        },
        "step_wise": {}, "autocorrelation": {}, "co_monotonicity": {},
    }

    for key, s in series.items():
        res["step_wise"][key] = {}
        for ca, cb, pair in PAIRS:
            pt = point_corr(s[ca], s[cb])
            ci = block_bootstrap_ci(s[ca], s[cb], block=block, nboot=args.nboot, rng=rng)
            res["step_wise"][key][pair] = {**pt, **ci}

    for col in ("index", WALK, FLOW):
        res["autocorrelation"][col] = integrated_autocorr_time(
            series[REF_KEY][col] if col == "index" else ref[col])

    for key, s in series.items():
        if key == REF_KEY:
            continue
        res["co_monotonicity"][key] = point_corr(ref["index"], s["index"])

    # Section 4.3: mask-vs-bbox gain in rho(Unrest Index, own walking distance),
    # classic Hausdorff, paired moving-block bootstrap at ~90 s blocks.
    block90 = max(4, int(round(90.0 * fs)))
    res["mask_vs_bbox_delta_rho"] = paired_delta_rho_ci(
        series["bbox_classic"]["index"], series["bbox_classic"][WALK],
        series["mask_classic"]["index"], series["mask_classic"][WALK],
        block=block90, nboot=args.nboot, rng=rng)
    res["mask_vs_bbox_delta_rho"]["block_sec"] = 90.0

    args.out_dir.joinpath("tables").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "correlations.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), "utf-8")
    emit_latex(res, args.out_dir / "tables" / "tab_correlacoes.tex")

    m = res["meta"]
    print(f"effective rate = {m['effective_rate_hz']:.2f} Hz | "
          f"bootstrap block = {m['bootstrap_block_steps']} steps (~{m['block_sec']:.0f} s)\n")
    for key, label in labels.items():
        print(f"[{label}]")
        for _, _, pair in PAIRS:
            d = res["step_wise"][key][pair]
            print(f"  {pair:34s} rho={d['spearman']:+.3f} "
                  f"[{d['spearman_ci'][0]:+.3f},{d['spearman_ci'][1]:+.3f}]  "
                  f"r={d['pearson']:+.3f}")
        print()
    dr = res["mask_vs_bbox_delta_rho"]
    print(f"delta-rho (mask - bbox, II x own walk, classic; {dr['block_steps']} steps"
          f" ~{dr['block_sec']:.0f} s blocks) = {dr['delta_rho']:+.3f} "
          f"[{dr['ci'][0]:+.3f}, {dr['ci'][1]:+.3f}]")
    print(f"\nwrote: {args.out_dir/'correlations.json'}\n"
          f"       {args.out_dir/'tables'/'tab_correlacoes.tex'}")


if __name__ == "__main__":
    main()
