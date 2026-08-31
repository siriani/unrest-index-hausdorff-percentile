#!/usr/bin/env python3
"""
analyze_replication.py -- replication of the ordering result across eight
adult-broiler videos, four of which contributed no frame to model training.

Reads, for each video and centroid source:
    <data-dir>/<video>_sweep_bbox/unrest_timeseries.csv
    <data-dir>/<video>_sweep_mask_v3/unrest_timeseries.csv

For the classic (max-based) Hausdorff distance it computes the step-wise
Spearman rho of the Unrest Index against walking distance and against optical
flow, plus the rho between the two reference metrics (flow-walk), which
indicates how consistent the optical-flow reference is in that video.

Key finding: with the mask centroid and the classic Hausdorff, rho(Unrest
Index x walking distance) = 0.77-0.89 across all eight videos, with no
difference between videos seen and not seen in training -- the ordering is
not an artefact of data leakage.

Outputs:
    <out-dir>/replication.json
    <out-dir>/tables/tab_replicacao.tex        (LaTeX fragment, pt-BR)
    <out-dir>/figures/replicacao_percentil.png

Usage:
    python3 src/analyze_replication.py [--data-dir data/timeseries] [--out-dir results]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (folder id, short label, in training pool?, annotated frames from this video)
VIDEOS = [
    # --- frames present in the training pool ---
    ("DVR-Ch1-CAM2-2026-06-18",          "DVR-CAM2 (principal)", True,  532),
    ("RecordPlan_20260423103037_1001_4", "RecordPlan-04/23",     True,  670),
    ("RecordPlan_20260224100150_1007_2", "RecordPlan-02/24a",    True,  191),
    ("RecordPlan_20260224140202_1007_3", "RecordPlan-02/24b",    True,  135),
    # --- zero frames in the training pool ---
    ("RecordPlan_20260409084432_1007_3", "RecordPlan-04/09",     False, 0),
    ("SimPlay-LCT2-CAM2-2026-03-16",     "SimPlay-CAM2",         False, 0),
    ("DVR-Ch2-CAM1-2026-06-18",          "DVR-Ch2 (outra baia)", False, 0),
    ("DVR-Ch3-CAM3-2026-06-18",          "DVR-Ch3 (outra baia)", False, 0),
]
K_COLS = [("unrest_p30", 30), ("unrest_p40", 40), ("unrest_p50", 50), ("unrest_p60", 60),
          ("unrest_p70", 70), ("unrest_p80", 80), ("unrest_p90", 90), ("unrest_p95", 95),
          ("unrest_classic", 100)]
KREC = 100  # 100 = classic Hausdorff (max); the operating point for ordering
WALK = "flock_walking_distance_step_px"
FLOW = "optical_flow_mean"


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
            pass
    return out


def rho(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    return float(stats.spearmanr(a, b)[0]) if a.size >= 10 else float("nan")


def col_for_k(k: int) -> str:
    return "unrest_classic" if k == 100 else f"unrest_p{k}"


def fmt(x, d=2):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}".replace(".", ",")


def emit_table(res: dict, out: Path) -> None:
    L = [
        "% Gerado por src/analyze_replication.py -- nao editar a mao.",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Replicação em oito vídeos de aves adultas, quatro deles ($^{\\dagger}$) "
        "sem nenhum quadro no pool de treino dos modelos. Correlação de postos de Spearman "
        "($\\rho$), passo a passo, do Unrest Index (Hausdorff clássica) com a distância "
        "percorrida, para as duas fontes de centroide. \\emph{flow--walk}: $\\rho$ entre "
        "as duas métricas de refer\\^encia (indica quão consistente é o \\emph{optical "
        "flow} naquele vídeo).}",
        "  \\label{tab:replicacao}",
        "  \\footnotesize\\setlength{\\tabcolsep}{5pt}",
        "  \\begin{tabular}{llccc}",
        "    \\toprule",
        "    Vídeo & fonte & \\emph{flow--walk} $\\rho$ & II $\\times$ dist.\\ $\\rho$ & II $\\times$ flow $\\rho$ \\\\",
        "    \\midrule",
    ]
    for vid, lbl, inpool, _ in VIDEOS:
        tag = lbl + ("" if inpool else "$^{\\dagger}$")
        first = True
        for src in ("bbox", "mask"):
            d = res["videos"][vid]["src"].get(src)
            if d is None:
                continue
            name = tag if first else ""
            srcname = "caixa" if src == "bbox" else "máscara"
            L.append(f"    {name} & {srcname} & {fmt(d['flow_walk_rho'])} & "
                     f"{fmt(d['index_walk_rho'])} & {fmt(d['index_flow_rho'])} \\\\")
            first = False
        L.append("    \\midrule")
    L[-1] = "    \\bottomrule"
    L += ["  \\end{tabular}", "\\end{table}",
          "% $^{\\dagger}$ sem nenhum quadro no pool de treino."]
    out.write_text("\n".join(L) + "\n", "utf-8")


def emit_figure(res: dict, out: Path) -> None:
    ncol = 4
    nrow = (len(VIDEOS) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3.0 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    xs = [k for _, k in K_COLS]
    xt = [("máx" if k == 100 else str(k)) for k in xs]
    for ax, (vid, lbl, inpool, _) in zip(axes, VIDEOS):
        d = res["videos"][vid]["src"].get("mask")
        if d is None:
            ax.set_visible(False)
            continue
        ax.plot(xs, [d["sweep"][str(k)]["walk"] for k in xs], "-o", ms=3, lw=1.4,
                color="#1b6ca8", label=r"$\rho$ II$\times$dist. percorrida")
        ax.plot(xs, [d["sweep"][str(k)]["flow"] for k in xs], "-^", ms=3, lw=1.2,
                color="#c1440e", label=r"$\rho$ II$\times$optical flow")
        ttl = lbl + ("" if inpool else " *")
        ax.set_title(f"{ttl}\n(flow–walk $\\rho$={d['flow_walk_rho']:.2f})", fontsize=8)
        ax.set_xticks(xs); ax.set_xticklabels(xt, fontsize=6.5, rotation=45)
        ax.set_ylim(-0.1, 1.0); ax.grid(alpha=0.3, lw=0.4)
        ax.axvspan(90, 100, color="#cfe8cf", alpha=0.5)
    for ax in axes[len(VIDEOS):]:
        ax.set_visible(False)
    for r in range(nrow):
        axes[r * ncol].set_ylabel(r"$\rho$")
    axes[min(len(VIDEOS), len(axes)) - 1].legend(fontsize=6.5, loc="lower right")
    fig.suptitle("Centroide de máscara --- $\\rho$ da varredura de percentil em oito vídeos "
                 "(* = sem quadros no treino; faixa: p90--máx, onde p95 $\\approx$ máx)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/timeseries"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    res: dict = {"meta": {"operating_k": KREC}, "videos": {}}
    for vid, lbl, inpool, nfr in VIDEOS:
        entry = {"label": lbl, "in_training_pool": inpool, "annotated_frames": nfr, "src": {}}
        for src, folder_src in (("bbox", "bbox"), ("mask", "mask_v3")):
            base = args.data_dir / f"{vid}_sweep_{folder_src}" / "unrest_timeseries.csv"
            path = base if base.exists() else base.with_suffix(".csv.gz")
            if not path.exists():
                print("MISSING", base, "(.csv or .csv.gz)")
                continue
            s = load_csv(path)
            nb = s["n_birds_detected"][np.isfinite(s["n_birds_detected"])]
            sweep = {str(k): {"walk": rho(s[col_for_k(k)], s[WALK]),
                              "flow": rho(s[col_for_k(k)], s[FLOW])}
                     for _, k in K_COLS}
            entry["src"][src] = {
                "n_birds_median": float(np.median(nb)),
                "steps": int(np.isfinite(s["unrest_classic"]).sum()),
                "index_walk_rho": sweep[str(KREC)]["walk"],
                "index_flow_rho": sweep[str(KREC)]["flow"],
                "flow_walk_rho": rho(s[FLOW], s[WALK]),
                "sweep": sweep,
            }
        res["videos"][vid] = entry

    args.out_dir.joinpath("tables").mkdir(parents=True, exist_ok=True)
    args.out_dir.joinpath("figures").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "replication.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), "utf-8")
    emit_table(res, args.out_dir / "tables" / "tab_replicacao.tex")
    emit_figure(res, args.out_dir / "figures" / "replicacao_percentil.png")

    print(f"{'video':26s} {'pool':5s} {'src':5s} {'nbirds':7s} {'II-walk rho':11s} {'II-flow rho':11s}")
    for vid, lbl, inpool, _ in VIDEOS:
        for src in ("bbox", "mask"):
            d = res["videos"][vid]["src"].get(src)
            if d is None:
                continue
            print(f"{lbl:26s} {str(inpool):5s} {src:5s} "
                  f"{d['n_birds_median']:7.0f} {d['index_walk_rho']:+.3f}      {d['index_flow_rho']:+.3f}")
    print(f"\nwrote: {args.out_dir/'replication.json'}\n"
          f"       {args.out_dir/'tables'/'tab_replicacao.tex'}\n"
          f"       {args.out_dir/'figures'/'replicacao_percentil.png'}")


if __name__ == "__main__":
    main()
