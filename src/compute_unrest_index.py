#!/usr/bin/env python3
"""
compute_unrest_index.py -- Unrest Index + walking distance + optical flow
for the Chicken_Unrest project (YOLO26m "ave" adult-chicken detector).

Formulas implemented faithfully from:
  - Del Valle et al. 2021, "Unrest index for estimating thermal comfort of
    poultry birds using computer vision techniques", Biosystems Engineering
    206:123-134. Eq. (6)-(7): Hausdorff distance; Eq. (12): Unrest Index.
    Eq. (8)-(11): pixel->cm calibration k = 2*H*tan(fov/2)/w -- NOT applied
    by default (no calibration data available); kept as an optional CLI
    parameter for future use once camera height/FOV are known.
  - animals-15-01311 (2025), "Equivalence Between Optical Flow, the Unrest
    Index, and Walking Distance to Estimate the Welfare of Broiler Chickens":
    per-ID cumulative walking distance from YOLO centroids (no persistent ID
    across the whole video required -- ID churn is tolerated), and 8x8-pixel
    block Farneback optical-flow moments (mean/var/skew/kurtosis).

Deviations/additions beyond the two papers are flagged inline as "IMPROVEMENT:".

INNOVATIONS (--centroid-source mask, --hausdorff-percentile):
  - Del Valle et al. 2021's centroid (Eq. 3) is the mean position of a bird's
    *segmented pixel blob* (classical background subtraction + Otsu
    threshold + morphology on a static, clean background). animals-15-01311
    (2025) approximates this instead with a YOLO *bounding-box* center, which
    is only a proxy for the true body centroid -- it is biased for an
    elongated/asymmetric pose (e.g. neck extended) and doesn't survive
    partial occlusion or cluttered backgrounds the way a real segmentation
    does. This script's innovation: use our OWN trained YOLO26-seg "ave"
    instance-segmentation model to recover a true per-bird pixel-area
    centroid (via cv2.moments on each predicted mask polygon) -- i.e. modern
    deep instance segmentation standing in for classical background
    subtraction, while staying literally faithful to Eq. 3's definition
    (mean occupied-area position), rather than the bbox-center approximation.
    Selected via --centroid-source {bbox,mask}; bbox is kept as the original
    (faster, simpler) baseline for comparison.
  - The classic Hausdorff distance (Eq. 6-7, a sup/max over points) is a
    worst-case statistic: a single missed or spurious detection in one frame
    can dominate the whole per-frame Unrest Index, regardless of how many
    other birds didn't move at all. This script adds an optional percentile
    ("partial") Hausdorff distance (Huttenlocher & Rucklidge 1993; Dubuisson
    & Jain 1994) -- replacing the max with the k-th percentile of the
    per-point nearest-neighbor distances -- which trades a small amount of
    worst-case sensitivity for much better robustness to single-frame
    detector noise. Selected via --hausdorff-percentile (None = classic
    Eq. 6-7 behaviour; e.g. 95 = 95th-percentile partial Hausdorff).
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BLOCK_SIZE = 8


def hausdorff_distance(a: np.ndarray, b: np.ndarray, percentile: float | None = None) -> float:
    """Symmetric Hausdorff distance between two 2D point sets (Eq. 6-7).
    Brute-force O(N*M) via numpy broadcasting -- no scipy dependency, since
    this project's venv pins numpy==1.26.4 and the skill notes warn not to
    disturb it with new heavy deps.

    INNOVATION: if `percentile` is given (e.g. 95), each directed distance
    uses the k-th percentile of per-point nearest-neighbor distances instead
    of the max/sup -- the "partial Hausdorff distance" (Huttenlocher &
    Rucklidge 1993; Dubuisson & Jain 1994), which is far less sensitive to a
    single missed/spurious detection than the classic max-based definition.
    percentile=None reproduces the exact classic Eq. 6-7 behaviour.
    """
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    nn_a_to_b = d.min(axis=1)  # for each point in a, distance to nearest point in b
    nn_b_to_a = d.min(axis=0)  # for each point in b, distance to nearest point in a
    if percentile is None:
        h_ab, h_ba = nn_a_to_b.max(), nn_b_to_a.max()
    else:
        h_ab, h_ba = np.percentile(nn_a_to_b, percentile), np.percentile(nn_b_to_a, percentile)
    return float(max(h_ab, h_ba))


def hausdorff_multi(a: np.ndarray, b: np.ndarray, percentiles) -> dict:
    """Symmetric Hausdorff distance for SEVERAL percentiles in ONE pass.

    The O(N*M) pairwise-distance matrix and the nearest-neighbour reductions
    are computed once; only the final percentile/max reduction repeats per
    k, so a full percentile sweep costs one detector inference pass instead
    of one per k. `percentiles` may contain None (=> classic max/sup).
    Returns {k: float}, with the key `None` for the classic variant.
    """
    if len(a) == 0 or len(b) == 0:
        return {k: float("nan") for k in percentiles}
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    nn_a_to_b = d.min(axis=1)
    nn_b_to_a = d.min(axis=0)
    out = {}
    for k in percentiles:
        if k is None:
            h_ab, h_ba = nn_a_to_b.max(), nn_b_to_a.max()
        else:
            h_ab = np.percentile(nn_a_to_b, k)
            h_ba = np.percentile(nn_b_to_a, k)
        out[k] = float(max(h_ab, h_ba))
    return out


def get_bbox_centroids(xyxy: np.ndarray) -> np.ndarray:
    """Bounding-box center -- the animals-15-01311 (2025) approximation."""
    if len(xyxy) == 0:
        return np.empty((0, 2))
    return np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2], axis=1)


def get_mask_centroids(masks_xy) -> np.ndarray:
    """INNOVATION: true per-instance pixel-area centroid from a segmentation
    polygon, via cv2.moments on the filled contour -- (m10/m00, m01/m00) is
    the exact area centroid of the occupied region, matching Del Valle et
    al. 2021 Eq. 3 ("mean position of the segmented foreground pixels")
    far more faithfully than a bounding-box center does.

    masks_xy: list of (N_i, 2) float arrays, one per detected instance
    (Ultralytics `Results.masks.xy`), pixel coordinates.
    """
    if not masks_xy:
        return np.empty((0, 2))
    centroids = []
    for poly in masks_xy:
        if poly is None or len(poly) < 3:
            continue
        contour = poly.astype(np.float32).reshape(-1, 1, 2)
        m = cv2.moments(contour)
        if m["m00"] > 1e-6:
            centroids.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))
        else:
            # IMPROVEMENT: degenerate (zero-area) polygon -- fall back to the
            # plain vertex mean rather than dropping the instance entirely.
            centroids.append((float(poly[:, 0].mean()), float(poly[:, 1].mean())))
    return np.array(centroids) if centroids else np.empty((0, 2))


def block_flow_moments(flow: np.ndarray, block_size: int = BLOCK_SIZE):
    """Average optical-flow magnitude per block_size x block_size block, then
    the four statistical moments (mean/var/skew/kurtosis) across all blocks."""
    mag = np.linalg.norm(flow, axis=-1)
    h, w = mag.shape
    hc, wc = (h // block_size) * block_size, (w // block_size) * block_size
    blocks = mag[:hc, :wc].reshape(hc // block_size, block_size, wc // block_size, block_size).mean(axis=(1, 3))
    v = blocks.ravel()
    mean, std = float(v.mean()), float(v.std())
    if std < 1e-9:
        # IMPROVEMENT: guard divide-by-zero on a fully static frame pair.
        return mean, 0.0, 0.0, 0.0
    skew = float(np.mean((v - mean) ** 3) / std ** 3)
    kurt = float(np.mean((v - mean) ** 4) / std ** 4 - 3.0)
    return mean, std ** 2, skew, kurt


class WalkingDistanceTracker:
    """Per-ID cumulative walking distance (sum of frame-to-frame centroid
    displacement, animals-15-01311 Sec 2.2). ID churn across the video is
    tolerated -- a new ID just starts its own accumulator at zero.

    IMPROVEMENT: a gap in a given ID's presence at consecutive *processed*
    steps (e.g. it dropped below --conf for one step) starts a fresh
    sub-segment instead of summing a long straight-line "teleport" across
    the gap, which would otherwise inflate distance with a detector/tracker
    artifact rather than real movement.
    """

    def __init__(self):
        self.last_step = {}
        self.last_pos = {}
        self.total = defaultdict(float)

    def update(self, step_idx, ids, positions):
        step_total = 0.0
        for tid, pos in zip(ids, positions):
            tid = int(tid)  # numpy int64 -> native int (JSON dict keys must be str/int/float/bool/None)
            if self.last_step.get(tid) == step_idx - 1:
                d = float(np.linalg.norm(pos - self.last_pos[tid]))
                self.total[tid] += d
                step_total += d
            self.last_step[tid] = step_idx
            self.last_pos[tid] = pos
        return step_total

    def flock_total(self):
        return sum(self.total.values())

    def per_id_summary(self):
        return dict(self.total)


def log_progress(step_idx, n_steps_expected, t0, n_flagged):
    elapsed = time.time() - t0
    rate = step_idx / elapsed if elapsed > 0 else 0.0
    eta = (n_steps_expected - step_idx) / rate if rate > 0 and n_steps_expected else float("nan")
    print(
        f"[{step_idx}/{n_steps_expected or '?'}] "
        f"{rate:.2f} steps/s, elapsed={elapsed:.1f}s, eta={eta:.1f}s, "
        f"flagged={n_flagged}",
        flush=True,
    )


def pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="Path to the source mp4 video.")
    ap.add_argument("--model", required=True, help="Path to the trained YOLO .pt weights (single class 'ave').")
    ap.add_argument("--output-dir", required=True, help="Directory to write CSV/JSON/PNG outputs into.")
    ap.add_argument("--conf", type=float, default=0.40, help="Detection confidence threshold.")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    ap.add_argument("--stride", type=int, default=3, help="Process every Nth native frame.")
    ap.add_argument("--flow-stride", type=int, default=1, help="Multiplier on --stride for optical-flow computation only.")
    ap.add_argument("--flow-scale", type=float, default=0.5, help="Downscale factor applied before Farneback optical flow.")
    ap.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config (bytetrack.yaml recommended for a static camera).")
    ap.add_argument("--device", default="0", help="Torch device for inference.")
    ap.add_argument(
        "--camera-height-cm",
        type=float,
        default=None,
        help=(
            "Camera height H (cm) above the floor. Combined with --fov-degrees, "
            "enables k = 2*H*tan(radians(fov)/2)/image_width_px per Del Valle et "
            "al. 2021 Eq. (8)-(11). Left as None: Unrest Index and walking "
            "distances are reported in raw pixels only (Eq. 12, uncalibrated)."
        ),
    )
    ap.add_argument("--fov-degrees", type=float, default=None, help="Camera lens horizontal FOV angle (degrees). See --camera-height-cm.")
    ap.add_argument("--max-seconds", type=float, default=None, help="Stop after this many seconds of video (for smoke tests).")
    ap.add_argument("--progress-every", type=int, default=200, help="Print a progress line every N processed steps.")
    ap.add_argument(
        "--centroid-source",
        choices=["bbox", "mask"],
        default="bbox",
        help=(
            "INNOVATION: 'bbox' (default) uses the YOLO bounding-box center, "
            "matching animals-15-01311 (2025). 'mask' requires --model to be a "
            "YOLO-SEG checkpoint and computes the true pixel-area centroid of "
            "each predicted instance mask (cv2.moments), matching Del Valle et "
            "al. 2021 Eq. 3 far more faithfully -- modern deep segmentation "
            "standing in for classical background-subtraction segmentation."
        ),
    )
    ap.add_argument(
        "--hausdorff-percentile",
        type=float,
        default=None,
        help=(
            "INNOVATION: if set (e.g. 95), use the partial/percentile Hausdorff "
            "distance (Huttenlocher & Rucklidge 1993; Dubuisson & Jain 1994) "
            "instead of the classic max-based Hausdorff (Eq. 6-7) -- much less "
            "sensitive to a single missed/spurious detection per frame. "
            "None = classic behaviour."
        ),
    )
    ap.add_argument(
        "--hausdorff-percentiles",
        default=None,
        help=(
            "SWEEP MODE: comma-separated list of percentiles to compute IN ONE "
            "PASS, e.g. '30,40,50,60,70,80,90,95'. Use 'classic' or 'max' in "
            "the list for the max-based variant. Each produces an extra CSV "
            "column unrest_pXX (and unrest_classic). The primary "
            "'unrest_index_px' column still follows --hausdorff-percentile. "
            "One detector inference pass covers the whole sweep."
        ),
    )
    args = ap.parse_args()

    sweep_ks = None
    if args.hausdorff_percentiles:
        sweep_ks = []
        for tok in args.hausdorff_percentiles.split(","):
            tok = tok.strip().lower()
            if not tok:
                continue
            sweep_ks.append(None if tok in ("classic", "max", "none") else float(tok))
        # stable, de-duplicated, None (classic) first
        seen, ordered = set(), []
        for k in ([None] if None in sweep_ks else []) + sorted(x for x in sweep_ks if x is not None):
            if k not in seen:
                seen.add(k)
                ordered.append(k)
        sweep_ks = ordered

    def _sweep_col(k):
        return "unrest_classic" if k is None else f"unrest_p{k:g}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model = YOLO(args.model)
    print(f"Model classes: {model.names}", flush=True)

    k_calibration = None
    if args.camera_height_cm is not None and args.fov_degrees is not None:
        k_calibration = 2 * args.camera_height_cm * np.tan(np.radians(args.fov_degrees) / 2) / width
        print(f"Calibration active: k={k_calibration:.6f} cm/px", flush=True)

    if args.max_seconds is not None:
        native_frames_to_process = min(native_total, int(args.max_seconds * fps))
    else:
        native_frames_to_process = native_total
    n_steps_expected = native_frames_to_process // args.stride

    wdt = WalkingDistanceTracker()
    prev_centroids = None
    prev_gray_small = None
    sweep_vals = {}
    step_idx = 0
    native_idx = 0
    n_flagged = 0
    t0 = time.time()
    min_conf_seen = None

    csv_path = out_dir / "unrest_timeseries.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    header = [
        "step_idx",
        "native_frame_idx",
        "timestamp_sec",
        "n_birds_detected",
        "unrest_index_px",
        "unrest_flag",
        "flock_walking_distance_step_px",
        "cumulative_flock_walking_distance_px",
        "optical_flow_mean",
        "optical_flow_var",
        "optical_flow_skew",
        "optical_flow_kurt",
    ]
    if k_calibration is not None:
        header += ["unrest_index_cm", "cumulative_flock_walking_distance_cm"]
    if sweep_ks is not None:
        header += [_sweep_col(k) for k in sweep_ks]
        print(f"Percentile sweep columns: {[_sweep_col(k) for k in sweep_ks]}", flush=True)
    writer.writerow(header)

    while True:
        if args.max_seconds is not None and native_idx / fps > args.max_seconds:
            break
        if native_idx % args.stride != 0:
            if not cap.grab():
                break
            native_idx += 1
            continue
        if not cap.grab():
            break
        ok, frame = cap.retrieve()
        if not ok:
            break

        results = model.track(
            frame,
            persist=True,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            tracker=args.tracker,
            verbose=False,
        )
        r = results[0]

        if r.boxes is not None and len(r.boxes):
            confs = r.boxes.conf.cpu().numpy()
            local_min = float(confs.min())
            min_conf_seen = local_min if min_conf_seen is None else min(min_conf_seen, local_min)
            # IMPROVEMENT: defensive re-filter -- ByteTrack keeps a lower internal
            # threshold to recover boxes for existing tracks; re-enforce --conf here
            # so the scientific centroid set is honest regardless of tracker internals.
            keep = confs >= args.conf
            xyxy = r.boxes.xyxy.cpu().numpy()[keep]
            ids = r.boxes.id.cpu().numpy().astype(int)[keep] if r.boxes.id is not None else np.array([], dtype=int)
            if args.centroid_source == "mask":
                if r.masks is None:
                    raise SystemExit(
                        "--centroid-source mask requires a YOLO-SEG checkpoint "
                        "(this model's results have no .masks -- did you pass a "
                        "detect-only .pt?)"
                    )
                masks_xy_all = r.masks.xy
                masks_xy = [poly for poly, k in zip(masks_xy_all, keep) if k]
                centroids = get_mask_centroids(masks_xy)
                # keep = boxes.conf filter may reorder relative to masks_xy_all only
                # if some were dropped; both come from the same per-instance axis
                # in Ultralytics' Results, so index alignment holds.
            else:
                centroids = get_bbox_centroids(xyxy)
        else:
            xyxy, ids, centroids = np.empty((0, 4)), np.array([], dtype=int), np.empty((0, 2))

        n_birds = len(centroids)

        if prev_centroids is None:
            unrest, flag = float("nan"), "first_step"
        elif n_birds == 0 and len(prev_centroids) == 0:
            unrest, flag = float("nan"), "both_empty"
        elif n_birds == 0 or len(prev_centroids) == 0:
            unrest, flag = float("nan"), "partial_empty"
        else:
            unrest, flag = hausdorff_distance(centroids, prev_centroids, args.hausdorff_percentile), "ok"
        if flag != "ok":
            n_flagged += 1

        if sweep_ks is not None:
            if flag == "ok":
                sweep_vals = hausdorff_multi(centroids, prev_centroids, sweep_ks)
            else:
                sweep_vals = {k: float("nan") for k in sweep_ks}

        step_walk = wdt.update(step_idx, ids, centroids) if len(ids) else 0.0
        cum_walk = wdt.flock_total()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if args.flow_scale != 1.0:
            small = cv2.resize(gray, None, fx=args.flow_scale, fy=args.flow_scale, interpolation=cv2.INTER_AREA)
        else:
            small = gray

        if prev_gray_small is None or step_idx % args.flow_stride != 0:
            of_mean = of_var = of_skew = of_kurt = float("nan")
        else:
            flow = cv2.calcOpticalFlowFarneback(prev_gray_small, small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            of_mean, of_var, of_skew, of_kurt = block_flow_moments(flow)
        prev_gray_small = small

        row = [
            step_idx,
            native_idx,
            native_idx / fps,
            n_birds,
            unrest,
            flag,
            step_walk,
            cum_walk,
            of_mean,
            of_var,
            of_skew,
            of_kurt,
        ]
        if k_calibration is not None:
            unrest_cm = unrest * k_calibration if not np.isnan(unrest) else float("nan")
            cum_walk_cm = cum_walk * k_calibration
            row += [unrest_cm, cum_walk_cm]
        if sweep_ks is not None:
            row += [sweep_vals[k] for k in sweep_ks]
        writer.writerow(row)

        prev_centroids = centroids
        step_idx += 1
        native_idx += 1

        if step_idx % args.progress_every == 0:
            log_progress(step_idx, n_steps_expected, t0, n_flagged)

    csv_file.close()
    cap.release()

    total_elapsed = time.time() - t0
    print(
        f"Done: {step_idx} steps processed in {total_elapsed:.1f}s "
        f"({step_idx / total_elapsed:.2f} steps/s). "
        f"Flagged (non-'ok') steps: {n_flagged}/{step_idx} "
        f"({100 * n_flagged / max(step_idx, 1):.2f}%). "
        f"Min detection confidence observed: {min_conf_seen}",
        flush=True,
    )

    # --- per-ID walking distance summary ---
    per_id = wdt.per_id_summary()
    with open(out_dir / "walking_distance_summary.json", "w") as f:
        json.dump(
            {
                "n_ids": len(per_id),
                "flock_total_px": wdt.flock_total(),
                "per_id_px": per_id,
            },
            f,
            indent=2,
        )
    with open(out_dir / "walking_distance_per_id.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "cumulative_distance_px"])
        for tid, d in sorted(per_id.items(), key=lambda kv: -kv[1]):
            w.writerow([tid, d])

    # --- run config for reproducibility ---
    run_config = {
        "video": args.video,
        "video_fps": fps,
        "video_width": width,
        "video_height": height,
        "video_native_total_frames": native_total,
        "model": args.model,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "stride": args.stride,
        "flow_stride": args.flow_stride,
        "flow_scale": args.flow_scale,
        "tracker": args.tracker,
        "device": args.device,
        "centroid_source": args.centroid_source,
        "hausdorff_percentile": args.hausdorff_percentile,
        "hausdorff_percentiles_sweep": (
            None if sweep_ks is None
            else ["classic" if k is None else k for k in sweep_ks]
        ),
        "camera_height_cm": args.camera_height_cm,
        "fov_degrees": args.fov_degrees,
        "k_calibration_applied": k_calibration is not None,
        "k_calibration_value_cm_per_px": k_calibration,
        "max_seconds": args.max_seconds,
        "steps_processed": step_idx,
        "elapsed_seconds": total_elapsed,
        "steps_per_second": step_idx / total_elapsed if total_elapsed > 0 else None,
        "flagged_steps": n_flagged,
        "min_detection_confidence_observed": min_conf_seen,
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # --- reload the CSV for correlations + plots ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {h: [] for h in header}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for h in header:
                v = row[h]
                if h == "unrest_flag":
                    data[h].append(v)
                else:
                    data[h].append(float(v) if v not in ("", "nan") else float("nan"))

    ok_mask = np.array([f == "ok" for f in data["unrest_flag"]])
    timestamps = np.array(data["timestamp_sec"])
    unrest_arr = np.array(data["unrest_index_px"])
    of_mean_arr = np.array(data["optical_flow_mean"])
    step_walk_arr = np.array(data["flock_walking_distance_step_px"])

    correlations = {
        "unrest_vs_optical_flow_mean": pearson(unrest_arr[ok_mask], of_mean_arr[ok_mask]),
        "unrest_vs_step_walking_distance": pearson(unrest_arr[ok_mask], step_walk_arr[ok_mask]),
        "optical_flow_mean_vs_step_walking_distance": pearson(of_mean_arr[ok_mask], step_walk_arr[ok_mask]),
        "n_rows_used": int(ok_mask.sum()),
    }
    with open(out_dir / "correlations.json", "w") as f:
        json.dump(correlations, f, indent=2)
    print(f"Correlations: {correlations}", flush=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(timestamps, unrest_arr, linewidth=0.7, color="tab:red")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Unrest Index (px)")
    ax.set_title(f"Unrest Index over time -- {Path(args.video).name}")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_unrest_index_over_time.png", dpi=150)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(timestamps, unrest_arr, linewidth=0.7, color="tab:red", label="Unrest Index (px)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Unrest Index (px)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(timestamps, of_mean_arr, linewidth=0.7, color="tab:blue", alpha=0.7, label="Optical flow mean")
    ax2.set_ylabel("Optical flow mean magnitude", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    fig.suptitle(f"Unrest Index vs Optical Flow -- r={correlations['unrest_vs_optical_flow_mean']:.3f}")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_unrest_vs_optical_flow.png", dpi=150)
    plt.close(fig)

    print(f"All outputs written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
