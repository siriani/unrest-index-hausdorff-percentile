# Per-instance outputs (not committed)

`src/analyze_paired_centroids.py` reads one folder per recording from here.
The files (~600 MB) are archived on Zenodo as `outputs.zip`
([10.5281/zenodo.23960761](https://doi.org/10.5281/zenodo.23960761)):

```bash
unzip outputs.zip -d data/paired_tmp && mv data/paired_tmp/outputs/* data/paired/
```

Each `<video>/` folder, written by `src/dump_paired_instances.py`, holds:

| file | content |
|---|---|
| `instances.csv.gz` | one row per instance per processed step: `step_idx`, `native_frame_idx`, `track_id` (−1 = returned without an identifier), `conf`, tracker box `x1..y2`, raw detector box `raw_x1..raw_y2`, `raw_match`, mask area centroid `mask_cx`, `mask_cy`, `mask_area_px`, `mask_ok` |
| `steps.csv` | one row per processed step: timestamp, number of tracked and dropped instances, optical-flow mean, variance, skewness and kurtosis |
| `flow_grid.npz` | optical-flow magnitude on 16-pixel blocks of the half-resolution frame (float16), for offline region-of-interest analyses |
| `run_config.json` | model, tracker configuration, library versions and raw-box matching diagnostics |
| `det_v5_timeseries.csv` | the detection-model series of the same recording (also in `data/timeseries/`) |
