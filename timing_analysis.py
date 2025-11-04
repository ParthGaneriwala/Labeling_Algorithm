#!/usr/bin/env python3
"""
All-in-one timing & savings analysis for hybrid labeling telemetry logs.

What it computes (across 1+ JSON logs):
- Files analyzed
- Unique images processed
- Wall-clock time (sum of session durations)
- Average seconds per image (wall-clock basis) + throughput
- ETA for a target number of images (default: 250,000)
- Manual vs Automated time (category sums from events; avoids umbrella double-counting)
- Estimated *fully manual* time if every frame were done by hand
- Estimated time saved vs fully manual
- Optional per-event duration percentiles by category
- Optional CSV exports for per-file rows and a single aggregate row

USAGE
  python timing_all_in_one.py /path/to/session_*.json
  python timing_all_in_one.py /mnt/data/session_20251102_141512_a9ee06.json --target 250000
  python timing_all_in_one.py "D:/PhDWork/Labeling_Algorithm/logs/session_*.json" --csv out.csv --perfile out_perfile.csv

If your event names differ, tweak MANUAL_EVENTS / AUTO_EVENTS / EXCLUDE_FROM_CATEGORY_TOTALS below.
"""

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, Set, List, Optional

ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"
# --------- Configure event -> category mappings --------------------------------
# User-driven work:
MANUAL_EVENTS = {
    "manual_annotation",
    "anchor_frame",  # selecting/setting anchors is manual in practice
}

# Automated processing / IO:
AUTO_EVENTS = {
    "propagate_annotations",
    "propagate_step",
    "propagate_annotations:drift",
    "generate_segmentation_masks",
    "generate_segmentation_masks:image",
    "write_frame_meta",
}

# Wrapper/umbrella events that overlap sub-steps; exclude from category sums
# to avoid double-counting. (Wall time still captures end-to-end.)
EXCLUDE_FROM_CATEGORY_TOTALS = {
    "process_image_sequence"
}

# Events whose extras may contain image/file identifiers
IMAGE_ID_KEYS = ("file", "image", "from", "to", "frame")

# Timestamp format produced by the telemetry logger (UTC 'Z' suffix)
ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


# --------- Helpers --------------------------------------------------------------
def event_bounds_secs(events):
    """Return (min_start, max_end, duration_sec) across all events in a file."""
    starts, ends = [], []
    for ev in events:
        s = parse_iso(ev.get("t_start")); e = parse_iso(ev.get("t_end"))
        if s: starts.append(s)
        if e: ends.append(e)
    if not starts or not ends:
        return None, None, 0.0
    t0, t1 = min(starts), max(ends)
    return t0, t1, (t1 - t0).total_seconds()

def parse_iso(ts):
    if not isinstance(ts, str):
        return None
    try:
        return datetime.strptime(ts, ISO_FMT).replace(tzinfo=timezone.utc)
    except Exception:
        # Try truncated microseconds
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return None

def is_image_like(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    vlow = v.lower()
    return vlow.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"))

def fmt_dur(s: float) -> str:
    if s < 60:
        return f"{s:.2f} s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{int(m)} min {s:.1f} s"
    h, m = divmod(m, 60)
    if h < 48:
        return f"{int(h)} h {int(m)} m"
    d, h = divmod(h, 24)
    return f"{int(d)} d {int(h)} h"


# --------- Core parsing ---------------------------------------------------------
def collect_from_file(fp: Path) -> Dict[str, Any]:
    with fp.open("r", encoding="utf-8") as f:
        blob = json.load(f)

    # Event-window & active-seconds
    t0, t1, event_window = event_bounds_secs(blob.get("events", []))
    active_secs = sum(float(ev.get("secs", 0.0)) for ev in blob.get("events", []))

    # Wall time: prefer session bounds; fall back to event bounds
    started = parse_iso(blob.get("started_at"))
    ended   = parse_iso(blob.get("ended_at"))
    if not (started and ended):
        ev_starts, ev_ends = [], []
        for ev in blob.get("events", []):
            s = parse_iso(ev.get("t_start")); e = parse_iso(ev.get("t_end"))
            if s: ev_starts.append(s)
            if e: ev_ends.append(e)
        if ev_starts and ev_ends:
            started = min(ev_starts) if not started else started
            ended   = max(ev_ends)   if not ended   else ended
    wall_secs = (ended - started).total_seconds() if (started and ended) else 0.0

    images: Set[str] = set()
    manual_time = 0.0
    auto_time = 0.0
    manual_frames = 0
    manual_event_secs: List[float] = []
    auto_event_secs: List[float] = []

    for ev in blob.get("events", []):
        name = ev.get("name", "")
        secs = float(ev.get("secs", 0.0))
        extra = ev.get("extra", {}) or {}

        # collect image ids when present
        for k in IMAGE_ID_KEYS:
            v = extra.get(k)
            if is_image_like(v):
                images.add(v)

        # categorize times (avoid umbrellas)
        if name in EXCLUDE_FROM_CATEGORY_TOTALS:
            pass
        elif name in MANUAL_EVENTS:
            manual_time += secs
            manual_event_secs.append(secs)
        elif name in AUTO_EVENTS:
            auto_time += secs
            auto_event_secs.append(secs)

        if name == "manual_annotation":
            manual_frames += 1

    # <-- return AFTER finishing the loop
    return {
        "file": str(fp),
        "wall_secs": wall_secs,
        "event_window_secs": event_window,
        "active_secs": active_secs,
        "images": images,
        "manual_time": manual_time,
        "auto_time": auto_time,
        "manual_frames": manual_frames,
        "manual_event_secs": manual_event_secs,
        "auto_event_secs": auto_event_secs,
    }




# --------- Aggregation & reporting ---------------------------------------------
def summarize(globs: Iterable[str],
              target_images: int = 250_000,
              csv_path: Optional[str] = None,
              perfile_csv_path: Optional[str] = None,
              show_percentiles: bool = True) -> None:
    

    # Resolve globs
    files: List[Path] = []
    for g in globs:
        files.extend(Path().glob(g))
    files = [f for f in files if f.is_file()]
    if not files:
        raise SystemExit("No files matched. Provide one or more JSON paths/globs.")

    # Per-file results (also used for optional CSV)
    rows: List[Dict[str, Any]] = []

    # Aggregates
    total_wall = 0.0
    all_images: Set[str] = set()
    total_manual_time = 0.0
    total_auto_time = 0.0
    total_manual_frames = 0
    manual_event_secs_all: List[float] = []
    auto_event_secs_all: List[float] = []
    total_event_window = 0.0
    total_active = 0.0

    for fp in files:
        try:
            out = collect_from_file(fp)
        except Exception as e:
            print(f"[WARN] Skipping {fp}: {e}")
            continue

        rows.append({
            "file": out["file"],
            "wall_secs": out["wall_secs"],
            "unique_images": len(out["images"]),
            "manual_time_secs": out["manual_time"],
            "auto_time_secs": out["auto_time"],
            "manual_frames": out["manual_frames"],
        })

        total_wall += out["wall_secs"]
        all_images |= out["images"]
        total_manual_time += out["manual_time"]
        total_auto_time += out["auto_time"]
        total_manual_frames += out["manual_frames"]
        manual_event_secs_all.extend(out["manual_event_secs"])
        auto_event_secs_all.extend(out["auto_event_secs"])
        total_event_window += out.get("event_window_secs", 0.0)
        total_active += out.get("active_secs", 0.0)


    file_count = len(rows)
    n_images = len(all_images)
    per_image_wall, thr_wall, eta_wall = eta_tuple(total_wall, n_images, target_images)
    per_image_ev,   thr_ev,   eta_ev   = eta_tuple(total_event_window, n_images, target_images)
    per_image_act,  thr_act,  eta_act  = eta_tuple(total_active, n_images, target_images)

    # Throughput & ETA (wall-clock is the most realistic end-to-end metric)
    if n_images > 0 and total_wall > 0:
        per_image_sec = total_wall / n_images
        throughput = 1.0 / per_image_sec if per_image_sec > 0 else 0.0
        eta_total_sec = per_image_sec * target_images
    else:
        per_image_sec = 0.0
        throughput = 0.0
        eta_total_sec = 0.0

    # Fully-manual baseline (per-manual-frame time × total images)
    if total_manual_frames > 0:
        per_manual_frame_sec = total_manual_time / total_manual_frames
    else:
        per_manual_frame_sec = 0.0
    fully_manual_est_sec = per_manual_frame_sec * n_images if n_images > 0 else 0.0

    # Estimated time saved vs fully manual (compare to actual wall time)
    savings_sec = max(0.0, fully_manual_est_sec - total_wall)
        # ---- Manual-only projection (anchors only) ----
    if n_images > 0:
        anchor_density = (total_manual_frames / n_images) if n_images > 0 else 0.0
        avg_manual_sec_per_frame = (total_manual_time / total_manual_frames) if total_manual_frames > 0 else 0.0
        manual_only_sec_per_image = (total_manual_time / n_images) if n_images > 0 else 0.0
        manual_only_thr = (1.0 / manual_only_sec_per_image) if manual_only_sec_per_image > 0 else 0.0
        manual_only_eta = manual_only_sec_per_image * target_images
    else:
        anchor_density = avg_manual_sec_per_frame = manual_only_sec_per_image = manual_only_thr = manual_only_eta = 0.0

    # ---- Print report ----
    print("==== Timing & Savings Summary ====")
    print(f"Files analyzed:                {file_count}")
    print(f"Unique images observed:        {n_images}")
    print(f"Wall time (sum):               {fmt_dur(total_wall)}")
    print(f"Avg time per image:            {per_image_sec:.6f} s (wall-clock)")
    print(f"Avg time per image (WALL):     {per_image_wall:.6f} s")
    print(f"Throughput (WALL):             {thr_wall:.3f} images/sec")
    print(f"ETA {target_images:,} (WALL):        {fmt_dur(eta_wall)}")
    print("---- Manual-only projection (anchors) ----")
    print(f"Anchor density:                {anchor_density*100:.2f}%  (= {total_manual_frames} / {n_images})")
    print(f"Manual sec per image:          {manual_only_sec_per_image:.6f} s")
    print(f"Throughput (manual-only):      {manual_only_thr:.3f} images/sec")
    print(f"ETA {target_images:,} (manual):      {fmt_dur(manual_only_eta)}\n")

    print()
    print(f"Avg time per image (EVENT):    {per_image_ev:.6f} s")
    print(f"Throughput (EVENT):            {thr_ev:.3f} images/sec")
    print(f"ETA {target_images:,} (EVENT):       {fmt_dur(eta_ev)}")
    print()
    print(f"Avg time per image (ACTIVE*):  {per_image_act:.6f} s")
    print(f"Throughput (ACTIVE*):          {thr_act:.3f} images/sec")
    print(f"ETA {target_images:,} (ACTIVE*):     {fmt_dur(eta_act)}")
    print("  *ACTIVE sums event durations (no overlap correction) — lower bound.\n")

    print()
    print("---- Breakdown (category sums; umbrella events excluded) ----")
    print(f"Manual time:                   {fmt_dur(total_manual_time)}")
    print(f"  Manual frames seen:          {total_manual_frames}")
    if total_manual_frames > 0:
        print(f"  Avg manual time/frame:       {per_manual_frame_sec:.4f} s")
    print(f"Automated processing time:     {fmt_dur(total_auto_time)}")
    if show_percentiles:
        def pctiles(arr: List[float]) -> str:
            if not arr:
                return "n/a"
            p50 = statistics.median(arr)
            # quantiles require n>=1; guard small samples
            p90 = statistics.quantiles(arr, n=10)[8] if len(arr) >= 10 else p50
            p99 = statistics.quantiles(arr, n=100)[98] if len(arr) >= 100 else p90
            return f"p50={p50:.4f}s  p90={p90:.4f}s  p99={p99:.4f}s"
        print(f"  Manual event percentiles:    {pctiles(manual_event_secs_all)}")
        print(f"  Auto event percentiles:      {pctiles(auto_event_secs_all)}")
    print()
    print("---- Automation benefit (vs fully manual) ----")
    print(f"Fully manual estimate:         {fmt_dur(fully_manual_est_sec)}  "
          f"(= {n_images} × {per_manual_frame_sec:.4f} s)")
    print(f"Actual wall-clock:             {fmt_dur(total_wall)}")
    print(f"Estimated time saved:          {fmt_dur(savings_sec)}")

    # ---- Optional CSVs ----
    if perfile_csv_path:
        with open(perfile_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    if csv_path:
        # One aggregate row
        agg = {
            "files_analyzed": file_count,
            "unique_images": n_images,
            "wall_secs": round(total_wall, 6),
            "avg_sec_per_image": round(per_image_sec, 6),
            "throughput_img_per_sec": round(throughput, 6),
            "eta_sec_for_target": round(eta_total_sec, 6),
            "manual_time_secs": round(total_manual_time, 6),
            "auto_time_secs": round(total_auto_time, 6),
            "manual_frames": total_manual_frames,
            "avg_manual_sec_per_frame": round(per_manual_frame_sec, 6),
            "fully_manual_est_sec": round(fully_manual_est_sec, 6),
            "time_saved_sec": round(savings_sec, 6),
        }
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(agg.keys()))
            w.writeheader()
            w.writerow(agg)

def eta_tuple(total_seconds, n_images, target_images):
    if n_images > 0 and total_seconds > 0:
        per_img = total_seconds / n_images
        thr = 1.0 / per_img
        eta = per_img * target_images
    else:
        per_img = thr = eta = 0.0
    return per_img, thr, eta


# --------- CLI ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="All-in-one timing & savings analysis for labeling logs.")
    ap.add_argument("paths", nargs="+", help="One or more JSON paths/globs (e.g., /mnt/data/session_*.json)")
    ap.add_argument("--target", type=int, default=250_000, help="Target number of images for ETA (default: 250000)")
    ap.add_argument("--csv", type=str, default=None, help="Write aggregate metrics to this CSV path")
    ap.add_argument("--perfile", type=str, default=None, help="Write per-file metrics to this CSV path")
    ap.add_argument("--no-pctiles", action="store_true", help="Disable percentile stats for event durations")
    args = ap.parse_args()

    summarize(
        args.paths,
        target_images=args.target,
        csv_path=args.csv,
        perfile_csv_path=args.perfile,
        show_percentiles=not args.no_pctiles
    )

if __name__ == "__main__":
    main()

# Aggregate all matching logs
# python timing_analysis.py D:\PhDWork\GX010009\annotations\logs\*.json

# Single file, with CSV outputs
# python timing_analysis.py D:\PhDWork\GX010009\annotations\logs\session_20251102_141512_a9ee06.json \
#   --target 250000 --csv agg.csv --perfile perfile.csv
