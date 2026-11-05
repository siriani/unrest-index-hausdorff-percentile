#!/usr/bin/env python3
"""
dump_paired_instances.py -- single segmentation pass per video, dumping every
tracked instance so that box-centre and mask-centroid variants of the Unrest
Index and of the tracker-derived travel distance can be computed OFFLINE from
IDENTICAL instances, frames, confidence threshold and track IDs.

Answers Animals-4688242 R1 (comments 3, 7, 16) and R3 (comment 1): the
original comparison used two separately trained models (det_v5 for boxes,
seg_v3 for masks), confounding the centroid definition with the detector.
Here only the segmentation model runs; the box centre and the mask centroid
are two read-outs of the same instance.

The tracking call is byte-for-byte the one used in src/compute_unrest_index.py
(model.track, persist=True, conf, imgsz, bytetrack.yaml, same stride), so the
mask-centroid series must reproduce the earlier *_sweep_mask_v3 run -- this is
checked offline as a sanity test.

Outputs in --output-dir:
  instances.csv.gz   one row per tracked instance per processed step:
                     step_idx, native_frame_idx, track_id, conf,
                     x1, y1, x2, y2, mask_cx, mask_cy, mask_area_px, mask_ok
  steps.csv          one row per processed step: step_idx, native_frame_idx,
                     timestamp_sec, n_tracked, optical-flow moments (8x8
                     blocks, identical to src/compute_unrest_index.py)
  flow_grid.npz      per-step block-mean flow magnitude on a coarse grid
                     (FLOW_GRID_BLOCK px blocks of the downscaled frame,
                     float16) -- lets a fixed region of interest be applied
                     offline (R1 comment 10, R3 comment 7) without re-running
  run_config.json    all parameters + tracker yaml + library versions
"""
import argparse
import csv
import gzip
import json
import time
from pathlib import Path

import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO

BLOCK_SIZE = 8          # same as src/compute_unrest_index.py
FLOW_GRID_BLOCK = 16    # coarse grid stored for offline ROI analysis


def block_flow_moments(mag: np.ndarray, block_size: int = BLOCK_SIZE):
    h, w = mag.shape
    hc, wc = (h // block_size) * block_size, (w // block_size) * block_size
    blocks = mag[:hc, :wc].reshape(hc // block_size, block_size, wc // block_size, block_size).mean(axis=(1, 3))
    v = blocks.ravel()
    mean, std = float(v.mean()), float(v.std())
    if std < 1e-9:
        return mean, 0.0, 0.0, 0.0
    skew = float(np.mean((v - mean) ** 3) / std ** 3)
    kurt = float(np.mean((v - mean) ** 4) / std ** 4 - 3.0)
    return mean, std ** 2, skew, kurt


def block_grid(mag: np.ndarray, block: int) -> np.ndarray:
    h, w = mag.shape
    hc, wc = (h // block) * block, (w // block) * block
    return mag[:hc, :wc].reshape(hc // block, block, wc // block, block).mean(axis=(1, 3))


def mask_centroid(poly):
    """Same definition as get_mask_centroids() in src/compute_unrest_index.py."""
    if poly is None or len(poly) < 3:
        return float("nan"), float("nan"), 0.0, 0
    contour = poly.astype(np.float32).reshape(-1, 1, 2)
    m = cv2.moments(contour)
    if m["m00"] > 1e-6:
        return m["m10"] / m["m00"], m["m01"] / m["m00"], float(m["m00"]), 1
    return float(poly[:, 0].mean()), float(poly[:, 1].mean()), 0.0, 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True, help="YOLO-SEG checkpoint")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--flow-scale", type=float, default=0.5)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--progress-every", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model = YOLO(args.model)
    print(f"Model classes: {model.names} task={model.task}", flush=True)
    if model.task != "segment":
        raise SystemExit("--model must be a segmentation checkpoint")

    # model.track() returns boxes taken from the ByteTrack Kalman-filter state
    # (STrack.tlwh uses self.mean), while masks are the raw detections indexed
    # by the track's det idx. To separate "centroid definition" from "Kalman
    # smoothing", stash the RAW detector boxes with a callback registered
    # BEFORE the tracker's own on_predict_postprocess_end (callbacks run in
    # registration order; the tracker registers on the first track() call).
    raw_stash = {}

    def _stash_raw(predictor):
        b = predictor.results[0].boxes
        raw_stash["data"] = b.data.cpu().numpy().copy() if b is not None and len(b) else np.empty((0, 6))

    model.add_callback("on_predict_postprocess_end", _stash_raw)

    inst_f = gzip.open(out / "instances.csv.gz", "wt", newline="")
    inst_w = csv.writer(inst_f)
    # x1..y2 = box returned by model.track() (Kalman posterior; the semantics of
    # the submitted manuscript); raw_x1..raw_y2 = detector box of the same
    # detection; mask_* from the same detection's mask. track_id = -1 marks a
    # detection the tracker returned without an ID (no active tracks).
    inst_w.writerow(["step_idx", "native_frame_idx", "track_id", "conf", "x1", "y1", "x2", "y2",
                     "raw_x1", "raw_y1", "raw_x2", "raw_y2", "raw_match",
                     "mask_cx", "mask_cy", "mask_area_px", "mask_ok"])
    n_ambiguous = n_idx_mismatch = 0
    track_high = None
    step_f = open(out / "steps.csv", "w", newline="")
    step_w = csv.writer(step_f)
    step_w.writerow(["step_idx", "native_frame_idx", "timestamp_sec", "n_tracked", "n_untracked_dropped",
                     "optical_flow_mean", "optical_flow_var", "optical_flow_skew", "optical_flow_kurt"])

    grids = []
    prev_small = None
    step_idx = native_idx = 0
    min_conf = None
    n_masks_missing = 0
    t0 = time.time()

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

        r = model.track(frame, persist=True, conf=args.conf, imgsz=args.imgsz, device=args.device,
                        tracker=args.tracker, verbose=False)[0]

        if track_high is None:
            track_high = float(model.predictor.trackers[0].args.track_high_thresh)
            # masks are aligned to tracks only if no detection is filtered out
            # inside ByteTrack before idx assignment (byte_tracker.update keeps
            # score >= track_high_thresh in the first stage); conf >= track_high
            # guarantees that, and also that no detection reaches the
            # low-confidence second association stage.
            assert args.conf >= track_high, (args.conf, track_high)

        raw = raw_stash.pop("data", np.empty((0, 6)))
        n_tracked = n_untracked = 0
        if r.boxes is not None and len(r.boxes):
            confs = r.boxes.conf.cpu().numpy()
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else None
            polys = r.masks.xy if r.masks is not None else [None] * len(confs)
            if r.masks is None:
                n_masks_missing += 1
            lc = float(confs.min())
            min_conf = lc if min_conf is None else min(min_conf, lc)
            if ids is not None:
                assert len(set(ids.tolist())) == len(ids), "duplicate track id in one step"
                # cross-check: det idx of output tracks, same comprehension the tracker uses
                tr = model.predictor.trackers[0]
                idx_internal = [int(t.idx) for t in tr.tracked_stracks if t.is_activated]
            for i in range(len(confs)):
                if ids is None:
                    # tracker returned raw detections (no active tracks): raw == returned box
                    j, match = i, 3
                else:
                    # the output track's score is the score of the detection matched this frame
                    cand = np.flatnonzero(np.abs(raw[:, 4] - confs[i]) < 1e-6)
                    if len(cand) == 1:
                        j, match = int(cand[0]), 1
                    else:
                        n_ambiguous += 1
                        j, match = (idx_internal[i], 2) if len(idx_internal) == len(confs) else (-1, 0)
                    if len(idx_internal) == len(confs) and idx_internal[i] != j:
                        n_idx_mismatch += 1
                rb = raw[j, :4] if j >= 0 else np.full(4, np.nan)
                cx, cy, area, mok = mask_centroid(polys[i])
                inst_w.writerow([step_idx, native_idx, -1 if ids is None else int(ids[i]), f"{confs[i]:.5f}",
                                 f"{xyxy[i, 0]:.3f}", f"{xyxy[i, 1]:.3f}", f"{xyxy[i, 2]:.3f}", f"{xyxy[i, 3]:.3f}",
                                 f"{rb[0]:.3f}", f"{rb[1]:.3f}", f"{rb[2]:.3f}", f"{rb[3]:.3f}", match,
                                 f"{cx:.3f}", f"{cy:.3f}", f"{area:.1f}", mok])
                if ids is None:
                    n_untracked += 1
                else:
                    n_tracked += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=args.flow_scale, fy=args.flow_scale, interpolation=cv2.INTER_AREA) \
            if args.flow_scale != 1.0 else gray
        if prev_small is None:
            fm = (float("nan"),) * 4
            grids.append(None)
        else:
            flow = cv2.calcOpticalFlowFarneback(prev_small, small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.linalg.norm(flow, axis=-1)
            fm = block_flow_moments(mag)
            grids.append(block_grid(mag, FLOW_GRID_BLOCK).astype(np.float16))
        prev_small = small

        step_w.writerow([step_idx, native_idx, f"{native_idx / fps:.4f}", n_tracked, n_untracked, *fm])
        step_idx += 1
        native_idx += 1
        if step_idx % args.progress_every == 0:
            el = time.time() - t0
            print(f"[{step_idx}] {step_idx / el:.2f} steps/s elapsed={el:.0f}s", flush=True)

    inst_f.close()
    step_f.close()
    cap.release()

    shape = next(g.shape for g in grids if g is not None)
    arr = np.stack([np.full(shape, np.nan, np.float16) if g is None else g for g in grids])
    np.savez_compressed(out / "flow_grid.npz", grid=arr, block_px_downscaled=FLOW_GRID_BLOCK,
                        flow_scale=args.flow_scale, frame_w=width, frame_h=height)

    import torch
    tracker_yaml = Path(ultralytics.__file__).parent / "cfg" / "trackers" / args.tracker
    cfg = {
        **vars(args),
        "video_fps": fps, "video_width": width, "video_height": height,
        "video_native_total_frames": native_total, "steps_processed": step_idx,
        "elapsed_seconds": time.time() - t0, "min_detection_confidence_observed": min_conf,
        "steps_with_boxes_but_no_masks": n_masks_missing,
        "raw_box_ambiguous_score_matches": n_ambiguous,
        "raw_box_idx_crosscheck_mismatches": n_idx_mismatch,
        "bytetrack_track_high_thresh": track_high,
        "ultralytics_version": ultralytics.__version__, "torch_version": torch.__version__,
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        "tracker_yaml": tracker_yaml.read_text() if tracker_yaml.exists() else None,
        "flow_block_moments_px": BLOCK_SIZE, "flow_grid_block_px_downscaled": FLOW_GRID_BLOCK,
    }
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"Done: {step_idx} steps in {time.time() - t0:.0f}s; min conf {min_conf}", flush=True)


if __name__ == "__main__":
    main()
