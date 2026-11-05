# Per-frame time series

One folder per **run** = one video processed with one centroid source. The
folder name is `<video>_sweep_<bbox|mask_v3>`. See `../MANIFEST.csv` for the
full list (video, resolution, stride, number of steps, and whether the video
contributed frames to the YOLO training pool).

Each folder holds:

| file | content |
|---|---|
| `unrest_timeseries.csv.gz` | one row per processed step (gzip-compressed) |
| `run_config.json` | exact parameters of the run (model, stride, conf, tracker, percentiles, resolution, timing) |
| `walking_distance_per_id.csv` | per-track cumulative walking distance |
| `walking_distance_summary.json` | flock totals and per-ID breakdown |

The analysis scripts (`src/analyze_*.py`) read `unrest_timeseries.csv.gz`
transparently; you do not need to decompress it.

## `unrest_timeseries.csv` columns

| column | meaning |
|---|---|
| `step_idx` | 0-based index of the processed step |
| `native_frame_idx` | frame index in the source video (`step_idx * stride`) |
| `timestamp_sec` | timestamp of the frame, from the video stream |
| `n_birds_detected` | number of birds in `F(i)` after the `conf` re-filter |
| `unrest_index_px` | Unrest Index for this step following `--hausdorff-percentile` (default = classic max), in pixels |
| `unrest_flag` | `ok`, `first_step`, `partial_empty` or `both_empty` |
| `flock_walking_distance_step_px` | flock walking distance **for this step** (per-step increment, summed over track IDs) |
| `cumulative_flock_walking_distance_px` | running sum of the above |
| `optical_flow_mean` | mean over 8x8 blocks of the mean Farneback flow magnitude |
| `optical_flow_var`, `optical_flow_skew`, `optical_flow_kurt` | higher moments of the same block field |
| `unrest_classic` | Unrest Index with the classic (max) Hausdorff distance |
| `unrest_p30 ... unrest_p95` | Unrest Index with the partial Hausdorff distance at percentile k (one inference pass, all k) |

Steps with `unrest_flag != ok` have `nan` in the Unrest Index columns and are
dropped from every correlation.

## Where the data lives

The `.csv.gz` files are committed here. The videos, annotations and
per-instance outputs are on Zenodo
([10.5281/zenodo.23960761](https://doi.org/10.5281/zenodo.23960761)). Cite the
Zenodo record for the data and the article for the method.
