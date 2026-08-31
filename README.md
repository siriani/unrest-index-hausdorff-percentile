# Calibrating the Unrest Index for adult broiler video

**The Hausdorff-distance percentile depends on the number of birds per frame.**

Code and derived time series for a step-wise, **rank-based** re-analysis of the
Unrest Index (Del Valle et al., 2021) on top-view video of adult broiler
chickens. This repository accompanies a manuscript in preparation (in
Portuguese; target journal *Computers and Electronics in Agriculture*,
Elsevier). Raw video and model weights are **not** here — see
[Data](#data-not-in-git).

## What the study does

The Unrest Index is the Hausdorff distance between the sets of per-frame bird
centroids in consecutive sampled frames. We re-implement it and evaluate two
modifications, judging each only by how well it **orders** the flock's activity
over time (Spearman rho against an independent walking-distance signal from
ByteTrack, and against Farneback dense optical flow):

1. **Mask-area centroid.** Replacing the bounding-box centre with the pixel-area
   centroid of a YOLO26 instance mask (`cv2.moments`) raises rho(index, walking
   distance) from ~0.80 to ~0.89 and cuts identity fragmentation.
2. **Partial (percentile) Hausdorff distance.** Replacing the max with the k-th
   percentile of nearest-neighbour distances — the HD95 idea from medical image
   segmentation. At ~10 birds per frame it does **not** help ordering: p95 is
   indistinguishable from the max, and aggressive percentiles (p50–p70) *lose*
   ordering fidelity.

The central result is the coupling between the useful percentile *k* and the
number of birds per frame *n*: **k\*(n) ≈ 100·(1 − 1/n)**. The percentile
window only opens up for larger flocks. The rho ≈ 0.8–0.9 agreement with
walking distance replicates step-by-step across 8 videos, including 4 with
**zero** frames in the YOLO training pool.

## Repository layout

```
.
├── src/
│   ├── compute_unrest_index.py   # video -> per-frame CSV (Unrest Index, walking distance, optical flow)
│   ├── analyze_correlations.py    # 4 configs on the main video -> Table 5 (tab_correlacoes.tex)
│   ├── analyze_percentile_sweep.py# percentile sweep + batch size -> Table 6 + Fig 4
│   └── analyze_replication.py     # 8-video replication -> Table 7 + Fig 5
├── data/
│   ├── MANIFEST.csv               # the 16 runs: video, centroid source, resolution, stride, steps, training frames
│   └── timeseries/<run>/          # one folder per (video x centroid source); see data/timeseries/README.md
│       ├── unrest_timeseries.csv.gz
│       ├── run_config.json
│       ├── walking_distance_per_id.csv
│       └── walking_distance_summary.json
├── results/
│   ├── correlations.json  percentile_sweep.json  replication.json
│   ├── tables/*.tex               # LaTeX fragments consumed by the manuscript (pt-BR strings on purpose)
│   └── figures/*.png
├── docs/REPRODUCIBILITY.md        # environment, training/annotation/inference hyperparameters
├── requirements.txt
├── LICENSE                        # MIT (code) + CC BY 4.0 (derived data under data/ and results/)
└── CITATION.cff
```

## Quick start (reproduce the tables and figures)

The derived time series are committed (gzipped), so the analysis runs without a
GPU or any video:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib          # or: pip install -r requirements.txt

python3 src/analyze_replication.py       --data-dir data/timeseries --out-dir results
python3 src/analyze_percentile_sweep.py  --data-dir data/timeseries --out-dir results
python3 src/analyze_correlations.py      --data-dir data/timeseries --out-dir results
```

Each script rewrites its JSON in `results/`, its `.tex` fragment in
`results/tables/`, and (the last two) a PNG in `results/figures/`. All three use
a fixed `--seed 0` moving-block bootstrap for the confidence intervals, so runs
are reproducible; `--nboot` trades runtime for CI precision (defaults: 2000 for
correlations, 1000 for the sweep).

> The `.tex` files contain Portuguese strings and `siunitx` comma decimals
> because they are `\input` directly into the manuscript. The numbers are the
> product; the surrounding prose is not meant to be read standalone here.

## Recomputing the time series from video

Only needed if you have the source `.mp4` and a trained checkpoint (neither is
in this repository):

```bash
pip install -r requirements.txt   # adds opencv-python, ultralytics; install torch for your CUDA

python3 src/compute_unrest_index.py \
    --video   path/to/video.mp4 \
    --model   path/to/yolo26m-seg.pt \
    --output-dir data/timeseries/<run> \
    --stride 3 --conf 0.40 --imgsz 640 \
    --centroid-source mask \
    --hausdorff-percentiles 30,40,50,60,70,80,90,95
```

`--centroid-source {bbox,mask}` picks the centroid; `mask` needs a `-seg`
checkpoint. `--hausdorff-percentiles` computes every percentile column in a
single inference pass (`unrest_p30 … unrest_p95` plus `unrest_classic`). Full
parameter list and the values used in the paper are in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Data not in git

* **Per-frame time series** — committed here as `data/timeseries/<run>/unrest_timeseries.csv.gz`
  (column schema in [data/timeseries/README.md](data/timeseries/README.md)) and
  also archived, with a snapshot of this repository, on Zenodo. The DOI is added
  on article acceptance.
* **Raw video** (`.dav`/`.mp4`) and **model weights** (`.pt`) — not distributed
  in git (size, and the recordings are from a commercial facility). Archived on
  Zenodo with the data.

## License and citation

Code: MIT. Derived data under `data/` and `results/`: CC BY 4.0. See
[LICENSE](LICENSE).

If you use this repository, cite the accompanying article (details in
[CITATION.cff](CITATION.cff); full reference added on publication) and the
Zenodo data record.

## References

* Del Valle, J. E. et al. (2021). Unrest index for estimating thermal comfort of
  poultry birds using computer vision techniques. *Biosystems Engineering*,
  206, 123–134.
* Pereira, D. F. et al. (2025). Equivalence between optical flow, the Unrest
  Index, and walking distance to estimate the welfare of broiler chickens.
  *Animals*, 15, 1311.
* Huttenlocher, D. P. & Rucklidge, W. J. (1993); Dubuisson, M.-P. & Jain, A. K.
  (1994); Taha, A. A. & Hanbury, A. (2015) — partial / percentile Hausdorff
  distance.
