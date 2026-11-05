# Unrest Index, tracker-derived travel distance and optical flow in broiler video

Analysis code and derived time series for the article

> Hübner, I. C.; Siriani, A. L. R.; Hokumura Filho, C. N.; Mollo Neto, M.;
> Moura, D. J. de; Barros, J. de S. G. **Comparison of the Unrest Index with
> Tracker-Derived Travel Distance and Optical Flow in Broiler Videos.**
> *Animals* (MDPI), manuscript animals-4688242, under review (revised version).

| | |
|---|---|
| Code | this repository (MIT licence) |
| Data (videos, annotations, per-instance outputs) | Zenodo, DOI [10.5281/zenodo.23960761](https://doi.org/10.5281/zenodo.23960761) (CC BY 4.0; the DOI resolves once the record is published) |
| Release for the revised manuscript | tag `v2.0-revision` |
| Release for the submitted manuscript | tag `v1.0-submission` (commit `d36de64`) |

## Overview

The Unrest Index (Del Valle et al., 2021) summarises flock activity as the
Hausdorff distance between the sets of bird positions detected in two
consecutive sampled frames. This repository computes the index from overhead
video of small broiler groups and compares it, step by step, with two other
activity measures:

* **tracker-derived travel distance**: the sum, over track identifiers present
  in two consecutive steps, of each bird's displacement (ByteTrack;
  Zhang et al., 2022);
* **dense optical flow**: Farnebäck (2003) flow magnitude, the only comparator
  that does not depend on the detections.

Agreement is measured by Spearman's rank correlation, with confidence intervals
from a moving-block bootstrap (Künsch, 1989).

Two variants of the index are examined:

1. **Position read-out.** The centre of the detection box versus the area
   centroid of the instance mask. In the revised analysis the two read-outs are
   taken from the *same* instances, frames and track identifiers of a single
   segmentation-model pass, which isolates the effect of the read-out from that
   of the model (`src/dump_paired_instances.py`, `src/analyze_paired_centroids.py`).
2. **Partial Hausdorff distance.** The *k*-th percentile of nearest-neighbour
   distances instead of the maximum (Huttenlocher et al., 1993).

### Results of the revised analysis (eight recordings)

The numbers below are taken from `results/paired_centroids.md`.

* With instances held fixed, the choice between the mask centroid and the box
  centre showed **no consistent direction** in agreement with optical flow.
  Across recordings Δρ ranged from −0.048 to +0.051 (mean 0.000).
* The improvement reported in the submitted manuscript is reproduced when only
  the **model run** changes: segmentation model versus detection model, both
  read at the tracker box centre, gave Δρ vs optical flow of +0.042 to +0.120,
  with the interval above zero in 8/8 recordings. It disappears when only the
  read-out changes.
* Index–distance agreement is partly **circular**. Replacing the travel distance
  computed from the same coordinates with one computed from a different
  read-out lowers ρ by about 0.19 (range 0.15–0.23). Detections and identifiers
  remain shared, so this is a lower bound.
* The p95 partial Hausdorff distance gave a **small but consistent** increase over
  the maximum: Δρ +0.02 to +0.04, with the interval above zero in all eight
  recordings.

Point estimates and conclusions are unchanged with bootstrap blocks of 10 s, 300 s
and 600 s (`results/long_blocks/`). All intervals describe uncertainty *within*
each recording. The eight
recordings include same-day and same-flock sets (see `videos_metadata.csv` on
Zenodo) and are not a sample of independent flocks.

## Repository layout

```
.
├── src/
│   ├── compute_unrest_index.py      # video -> per-step series (index, travel distance, optical flow)
│   ├── dump_paired_instances.py     # video -> per-instance table (raw box, tracker box, mask centroid) [revision]
│   ├── analyze_paired_centroids.py  # controlled centroid / model / circularity contrasts [revision]
│   ├── analyze_flow_statistics.py   # agreement with the four optical-flow moments [revision]
│   ├── analyze_correlations.py      # principal recording, four configurations [submission]
│   ├── analyze_percentile_sweep.py  # percentile sweep [submission]
│   └── analyze_replication.py       # eight-recording replication [submission]
├── data/
│   ├── MANIFEST.csv                 # the 16 per-step runs (video x model/read-out)
│   ├── timeseries/<run>/            # per-step series, committed (see data/timeseries/README.md)
│   └── paired/README.md             # where to place the per-instance outputs from Zenodo
├── results/
│   ├── paired_centroids.{json,md}   # revision: paired bootstrap, blocks of 90 s and 10 s
│   ├── long_blocks/paired_centroids.{json,md}  # revision: blocks of 300 s and 600 s
│   ├── flow_statistics.{csv,md}     # revision: four optical-flow statistics
│   ├── *.json, tables/, figures/    # submission: Tables 5-7, Figures 4-5
├── docs/REPRODUCIBILITY.md          # environment, training, annotation, inference and analysis settings
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Reproducing the results

The analysis scripts need neither a GPU nor the videos.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # analysis only: numpy, scipy, pandas, matplotlib, tabulate

# Revision: four optical-flow statistics (seconds; uses the committed series)
python3 src/analyze_flow_statistics.py

# Revision: paired centroid analysis (needs outputs.zip from Zenodo)
unzip outputs.zip -d data/paired_tmp && mv data/paired_tmp/outputs/* data/paired/
python3 src/analyze_paired_centroids.py --nboot 2000 --seed 20261104 --blocks 90,10
python3 src/analyze_paired_centroids.py --nboot 2000 --seed 20261104 --blocks 300,600 \
        --out-dir results/long_blocks     # block-length sensitivity

# Submission: Tables 5-7 and Figures 4-5
python3 src/analyze_replication.py      --data-dir data/timeseries --out-dir results
python3 src/analyze_percentile_sweep.py --data-dir data/timeseries --out-dir results
python3 src/analyze_correlations.py     --data-dir data/timeseries --out-dir results
```

The paired analysis takes roughly 10–15 min per recording on one CPU core with
2000 replicates. Seeds are fixed, so reruns give identical output.

### Recomputing the series from video

This step needs the videos (Zenodo), a trained checkpoint, `opencv-python`,
`ultralytics==8.4.51` and a CUDA build of `torch`:

```bash
python3 src/compute_unrest_index.py --video video.mp4 --model seg.pt \
    --output-dir out/ --stride 3 --conf 0.40 --imgsz 640 \
    --centroid-source mask --hausdorff-percentiles 30,40,50,60,70,80,90,95

python3 src/dump_paired_instances.py --video video.mp4 --model seg.pt \
    --output-dir data/paired/<video>/ --stride 3 --conf 0.40 --imgsz 640
```

The parameter values used in the article are listed in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). The model weights are not
distributed. The annotated frames and their training/validation lists are on
Zenodo (`annotations.zip`), so the models can be retrained.

## Data

* **Per-step series** (`data/timeseries/`) are committed here, gzip-compressed.
* **Videos, manual annotations, per-instance outputs and recording metadata**
  are on Zenodo ([10.5281/zenodo.23960761](https://doi.org/10.5281/zenodo.23960761)):
  * `video_*.mp4`: the eight recordings;
  * `videos_metadata.csv`: verified date and time, batch, compartment,
    treatment, bird age, resolution and frame rates;
  * `annotations.zip`: YOLO datasets with the exact split lists;
  * `outputs.zip`: per-instance tables, optical-flow grids and run
    configurations.

The recordings were made at the Laboratory of Thermal Comfort II, School of
Agricultural Engineering, University of Campinas (FEAGRI/UNICAMP), under
protocol CEUA/UNICAMP 6519-1/2024. No intervention was made for this study.

## How to cite

Cite the article (reference above; full citation on publication) and, for the
data, the Zenodo record. Citation metadata for this repository is in
[CITATION.cff](CITATION.cff).

## Funding

São Paulo Research Foundation (FAPESP, grant 2024/21166-9) and the Brazilian
National Council for Scientific and Technological Development (CNPq, grants
443961/2024-7 and 308177/2021-5).

## Licence

Code: MIT. Derived data in `data/` and `results/`: CC BY 4.0. See
[LICENSE](LICENSE).

## References

* Del Valle, J. E.; Pereira, D. F.; Mollo Neto, M.; Gabriel Filho, L. R. A.;
  Salgado, D. D. (2021). Unrest index for estimating thermal comfort of poultry
  birds (*Gallus gallus domesticus*) using computer vision techniques.
  *Biosystems Engineering*, 206, 123–134.
  https://doi.org/10.1016/j.biosystemseng.2021.03.018
* Farnebäck, G. (2003). Two-frame motion estimation based on polynomial
  expansion. In *Image Analysis (SCIA 2003)*, LNCS 2749, 363–370.
  https://doi.org/10.1007/3-540-45103-X_50
* Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics*, 6, 65–70.
* Huttenlocher, D. P.; Klanderman, G. A.; Rucklidge, W. J. (1993). Comparing
  images using the Hausdorff distance. *IEEE Transactions on Pattern Analysis
  and Machine Intelligence*, 15(9), 850–863.
  https://doi.org/10.1109/34.232073
* Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary
  observations. *The Annals of Statistics*, 17(3), 1217–1241.
* Pereira, D. F.; Nääs, I. de A.; Mehdizadeh, S. A. (2025). Equivalence between
  optical flow, the Unrest Index, and walking distance to estimate the welfare
  of broiler chickens. *Animals*, 15, 1311. https://doi.org/10.3390/ani15091311
* Zhang, Y.; Sun, P.; Jiang, Y.; Yu, D.; Weng, F.; Yuan, Z.; Luo, P.; Liu, W.;
  Wang, X. (2022). ByteTrack: multi-object tracking by associating every
  detection box. In *Computer Vision – ECCV 2022*, 1–21.
  https://doi.org/10.1007/978-3-031-20047-2_1
