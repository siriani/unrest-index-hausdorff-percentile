# Reproducibility details

English version of the manuscript's reproducibility appendix. `[...]` marks
placeholders the authors will fill before submission.

## Environment

Linux server (Ubuntu 24.04), NVIDIA GeForce RTX 5060 Ti (16 GB, driver
590.48.01), CUDA 12.8. Libraries: Python 3, `numpy` 1.26.4, `opencv-python`
4.11.0, `torch` 2.11.0+cu128, `ultralytics` 8.4.51. The Hausdorff distances
and the optical-flow moments are computed with `numpy` broadcasting (no
`scipy`/`pandas` inside `compute_unrest_index.py`); the post-hoc analysis
(Spearman, bootstrap) uses `scipy` 1.17.

## Model training

Two YOLO26m models, single class `ave` (bird), COCO-initialised. Common
settings: `batch` 8, `imgsz` 640, `optimizer=auto`, `lr0 = lrf = 0.01`,
`momentum` 0.937, `weight_decay` 5e-4, `warmup_epochs` 3, `cos_lr` off,
`close_mosaic` 10, `seed` 0, `deterministic`, `amp`; default Ultralytics
augmentation (`hsv_h` 0.015, `hsv_s` 0.7, `hsv_v` 0.4, `translate` 0.1,
`scale` 0.5, `fliplr` 0.5, `mosaic` 1.0, `erasing` 0.4).

* **Detector**: 100 epochs.
* **Instance segmenter (`seg v3`)**: 150 epochs, `overlap_mask`, `mask_ratio` 4.

Frame-level train/validation split (0.8 / 0.2) over the combined 1528-frame
pool. Per-source-video frame counts (Table 1 of the paper): `RecordPlan_20260423103037`
670, `DVR-Ch.1-CAM 2-2026-06-18` 532, `RecordPlan_20260224100150` 191,
`RecordPlan_20260224140202` 135. Split seeds: `[to be deposited]`.

Model weights are not in this repository (each ~40-50 MB); they are archived
on Zenodo with the data.

## Annotation

Tool: CVAT v2.73.0. Every bounding box and instance mask of the 1528 frames
was reviewed and corrected manually, frame by frame (100% of the set); SAM
3.1 was used only as a mask seed. `[Annotation team and, if applicable,
inter-annotator agreement.]`

### Segmentation-set correction (`seg v3`)

Earlier iterations prompted SAM 3.1 *inside* each detected bird's bounding-box
crop (points or text). Whole-frame pre-processing experiments (CLAHE, HSV
bird/litter enhancement, Otsu threshold, median-background suppression) did
not improve mask acceptance. The fix was simply to re-run the text prompt on
the *whole frame* (no crop): 9.94 masks/frame on average over the 1528
frames, up from 3.03/frame in the previous set; checked on a 30-frame random
sample (9.23 matched instances vs 9.50 detector boxes; 80% of frames with
>= 9). Additive merge into the CVAT task with an IoU < 0.3 criterion;
post-merge audit found zero same-image pairs with IoU >= 0.5. The corrected
set has 15 185 instances.

## Inference and tracking

`model.track` (Ultralytics) with `persist=True`, `conf=0.40`, `imgsz=640`,
`tracker="bytetrack.yaml"`; detections are re-filtered at `conf >= 0.40`
after the call (ByteTrack keeps a lower internal threshold to recover
track boxes). The `bytetrack.yaml` used is the Ultralytics 8.4.51 default:
`track_high_thresh` 0.25; `track_low_thresh` 0.10; `new_track_thresh` 0.25;
`match_thresh` 0.80; `track_buffer` 30; `fuse_score` True.

One frame is processed every 3 native frames (`stride = 3`).

## Optical flow

`cv2.calcOpticalFlowFarneback` with `pyr_scale` 0.5, `levels` 3, `winsize`
15, `iterations` 3, `poly_n` 5, `poly_sigma` 1.2, `flags` 0, on the frame
downscaled by 0.5x and converted to greyscale. 8x8 blocks; the reported
metric is the mean, over blocks, of the mean flow magnitude per block.

## Partial Hausdorff distance

Percentile via `numpy.percentile` (linear interpolation between order
statistics). Steps with an empty `F(i)` or `F(i-1)` are flagged and dropped
from every correlation (main video: 1 step, the first).

## Analysis

`src/analyze_*.py` read the per-frame CSVs and reproduce the paper's Tables
5-7 and Figures 4-5. Spearman rho is the reported statistic (the series are
heavy-tailed; Pearson is dominated by a handful of extreme steps and is kept
in Table 5 only for comparability with Pereira et al., 2025). CIs are from a
moving-block bootstrap (~10 s blocks); the mask-vs-bbox rho gain also gets a
paired moving-block bootstrap.
