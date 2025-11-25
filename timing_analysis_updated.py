
#!/usr/bin/env python3
"""
Timing & savings analysis for hybrid labeling telemetry logs.

Adds explicit breakdown:
- Manual time
- Automated propagation time (propagate_annotations)
- Segmentation masks time (generate_segmentation_masks:image)
- Other automated time (write_frame_meta, drift checks, etc.)
- Automated total

Also reports: files analyzed, unique images, wall-clock, per-image averages,
throughput, ETAs, manual-only projections, and optional CSV exports.

USAGE
  python timing_analysis.py /path/to/session_*.json
  python timing_analysis.py /mnt/data/session_20251103_191913_f4c84f.json /mnt/data/session_20251112_134644_82d732.json
  python timing_analysis.py "/logs/session_*.json" --csv agg.csv --perfile perfile.csv
"""

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, Set, List, Optional

# Timestamp format produced by the telemetry logger (UTC 'Z' suffix)
ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"

# --------- Configure event -> category mappings --------------------------------
# User-driven work:
MANUAL_EVENTS = {
    "manual_annotation",
    "anchor_frame",  # selecting/setting anchors is manual in practice
}

# Automated processing (we'll break this down explicitly below):
AUTO_PROPAGATE_EVENT = "propagate_annotations"             # main propagation work
AUTO_MASK_EVENT      = "generate_segmentation_masks:image" # per-image mask timers

# Minor/IO/heartbeat automation still counted as automated but not in the two key buckets
OTHER_AUTO_EVENTS = {
    "propagate_annotations:drift",
    "write_frame_meta",
}

# Wrapper/umbrella events that overlap sub-steps; exclude from category sums
EXCLUDE_FROM_CATEGORY_TOTALS = {
    "process_image_sequence",
    "anchor_frame",                    # wraps manual work and/or reuse
    "propagate_step",                  # wraps propagate_annotations(...)
    "generate_segmentation_masks",     # wraps per-image mask timers
    "generate_segmentation_masks:progress",  # heartbeat, secs=0 but be explicit
}

# Events whose extras may contain image/file identifiers
IMAGE_ID_KEYS = ("file", "image", "from", "to", "frame")


# --------- Helpers --------------------------------------------------------------
def parse_iso(ts: Optional[str]):
    if not isinstance(ts, str):
        return None
    try:
        return datetime.strptime(ts, ISO_FMT).replace(tzinfo=timezone.utc)
    except Exception:
        # Try more permissive parsing (handles truncated microseconds)
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return None

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
    auto_propagate_time = 0.0
    auto_masks_time = 0.0
    other_auto_time = 0.0
    manual_frames = 0
    manual_event_secs: List[float] = []
    auto_event_secs: List[float] = []  # all automated events combined
    propagate_event_secs: List[float] = []
    mask_event_secs: List[float] = []
    other_auto_event_secs: List[float] = []

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
        elif name == AUTO_PROPAGATE_EVENT:
            auto_propagate_time += secs
            auto_event_secs.append(secs)
            propagate_event_secs.append(secs)
        elif name == AUTO_MASK_EVENT:
            auto_masks_time += secs
            auto_event_secs.append(secs)
            mask_event_secs.append(secs)
        elif name in OTHER_AUTO_EVENTS:
            other_auto_time += secs
            auto_event_secs.append(secs)
            other_auto_event_secs.append(secs)
        # If there are unknown auto events in the future, we treat them as "other_auto"
        elif name.startswith("propagate_") or name.startswith("generate_") or name.startswith("write_"):
            other_auto_time += secs
            auto_event_secs.append(secs)
            other_auto_event_secs.append(secs)

        if name == "manual_annotation":
            manual_frames += 1

    auto_time_total = auto_propagate_time + auto_masks_time + other_auto_time

    return {
        "file": str(fp),
        "wall_secs": wall_secs,
        "event_window_secs": event_window,
        "active_secs": active_secs,
        "images": images,
        "manual_time": manual_time,
        "auto_propagate_time": auto_propagate_time,
        "auto_masks_time": auto_masks_time,
        "other_auto_time": other_auto_time,
        "auto_time_total": auto_time_total,
        "manual_frames": manual_frames,
        "manual_event_secs": manual_event_secs,
        "propagate_event_secs": propagate_event_secs,
        "mask_event_secs": mask_event_secs,
        "other_auto_event_secs": other_auto_event_secs,
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

    # Per-file rows (used for optional CSV)
    rows: List[Dict[str, Any]] = []

    # Aggregates
    total_wall = 0.0
    all_images: Set[str] = set()
    total_manual_time = 0.0
    total_auto_propagate = 0.0
    total_auto_masks = 0.0
    total_other_auto = 0.0
    total_auto_time = 0.0
    total_manual_frames = 0
    manual_event_secs_all: List[float] = []
    auto_event_secs_all: List[float] = []
    propagate_event_secs_all: List[float] = []
    mask_event_secs_all: List[float] = []
    other_auto_event_secs_all: List[float] = []
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
            "auto_propagate_secs": out["auto_propagate_time"],
            "auto_masks_secs": out["auto_masks_time"],
            "other_auto_secs": out["other_auto_time"],
            "auto_total_secs": out["auto_time_total"],
            "manual_frames": out["manual_frames"],
        })

        total_wall += out["wall_secs"]
        all_images |= out["images"]
        total_manual_time += out["manual_time"]
        total_auto_propagate += out["auto_propagate_time"]
        total_auto_masks += out["auto_masks_time"]
        total_other_auto += out["other_auto_time"]
        total_auto_time += out["auto_time_total"]
        total_manual_frames += out["manual_frames"]
        manual_event_secs_all.extend(out["manual_event_secs"])
        propagate_event_secs_all.extend(out.get("propagate_event_secs", []))
        mask_event_secs_all.extend(out.get("mask_event_secs", []))
        other_auto_event_secs_all.extend(out.get("other_auto_event_secs", []))
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
        anchor_density = (total_manual_frames / n_images)
        avg_manual_sec_per_frame = (total_manual_time / total_manual_frames) if total_manual_frames > 0 else 0.0
        manual_only_sec_per_image = (total_manual_time / n_images)
        manual_only_thr = (1.0 / manual_only_sec_per_image) if manual_only_sec_per_image > 0 else 0.0
        manual_only_eta = manual_only_sec_per_image * target_images
    else:
        anchor_density = avg_manual_sec_per_frame = manual_only_sec_per_image = manual_only_thr = manual_only_eta = 0.0

    # ---- Print report ----
    print("==== Timing & Savings Summary ====")
    print(f"Files analyzed:                        {file_count}")
    print(f"Unique images observed:                {n_images}")
    print(f"Wall time (sum):                       {fmt_dur(total_wall)}")
    print(f"Avg time per image:                    {per_image_sec:.6f} s (wall-clock)")
    print(f"Avg time per image (WALL):             {per_image_wall:.6f} s")
    print(f"Throughput (WALL):                     {thr_wall:.3f} images/sec")
    print(f"ETA {target_images:,} (WALL):                {fmt_dur(eta_wall)}")

    print("\n---- Manual-only projection (anchors) ----")
    print(f"Anchor density:                        {anchor_density*100:.2f}%  (= {total_manual_frames} / {n_images})")
    print(f"Manual sec per image:                  {manual_only_sec_per_image:.6f} s")
    print(f"Throughput (manual-only):              {manual_only_thr:.3f} images/sec")
    print(f"ETA {target_images:,} (manual):              {fmt_dur(manual_only_eta)}\n")

    print(f"Avg time per image (EVENT):            {per_image_ev:.6f} s")
    print(f"Throughput (EVENT):                    {thr_ev:.3f} images/sec")
    print(f"ETA {target_images:,} (EVENT):               {fmt_dur(eta_ev)}\n")

    print(f"Avg time per image (ACTIVE*):          {per_image_act:.6f} s")
    print(f"Throughput (ACTIVE*):                  {thr_act:.3f} images/sec")
    print(f"ETA {target_images:,} (ACTIVE*):             {fmt_dur(eta_act)}")
    print("  *ACTIVE sums event durations (no overlap correction) — upper bound (can exceed wall time).\n")


    print("---- Breakdown (category sums; umbrella events excluded) ----")
    print(f"Manual time:                           {fmt_dur(total_manual_time)}")
    if total_manual_frames > 0:
        print(f"  Manual frames seen:                  {total_manual_frames}")
        print(f"  Avg manual time/frame:               {per_manual_frame_sec:.4f} s")
    print(f"Automated propagation time:            {fmt_dur(total_auto_propagate)}")
    print(f"Segmentation masks time:               {fmt_dur(total_auto_masks)}")
    print(f"Other automated time:                  {fmt_dur(total_other_auto)}")
    print(f"Automated processing time (TOTAL):     {fmt_dur(total_auto_time)}")
    # Shares relative to wall-clock (unattributed covers overlaps/idle/umbrella exclusions)
    if total_wall > 0:
        share_manual = total_manual_time / total_wall
        share_prop   = total_auto_propagate / total_wall
        share_masks  = total_auto_masks / total_wall
        share_other  = total_other_auto / total_wall
        share_unattr = max(0.0, 1.0 - (share_manual + share_prop + share_masks + share_other))
        print("  Shares (of wall time):               "
              f"Manual {share_manual*100:.2f}%  |  Propagation {share_prop*100:.2f}%  |  "
              f"Masks {share_masks*100:.2f}%  |  Other {share_other*100:.2f}%  |  "
              f"Unattributed {share_unattr*100:.2f}%")

    if show_percentiles:
        def pctiles(arr: List[float]) -> str:
            if not arr:
                return "n/a"
            p50 = statistics.median(arr)
            # Guard small samples for quantiles
            p90 = statistics.quantiles(arr, n=10)[8] if len(arr) >= 10 else p50
            p99 = statistics.quantiles(arr, n=100)[98] if len(arr) >= 100 else p90
            return f"p50={p50:.4f}s  p90={p90:.4f}s  p99={p99:.4f}s"
        print(f"  Manual event percentiles:            {pctiles(manual_event_secs_all)}")
        print(f"  Auto event percentiles (all auto):   {pctiles(auto_event_secs_all)}")
        print(f"  Auto (propagation) percentiles:      {pctiles(propagate_event_secs_all)}")
        print(f"  Auto (masks) percentiles:            {pctiles(mask_event_secs_all)}")
        print(f"  Auto (other) percentiles:            {pctiles(other_auto_event_secs_all)}")
    print()
    print("---- Automation benefit (vs fully manual) ----")
    print(f"Fully manual estimate:                 {fmt_dur(fully_manual_est_sec)}  "
          f"(= {n_images} × {per_manual_frame_sec:.4f} s)")
    print(f"Actual wall-clock:                     {fmt_dur(total_wall)}")
    print(f"Estimated time saved:                  {fmt_dur(savings_sec)}")

    # ---- Optional CSVs ----
    if perfile_csv_path and rows:
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
            "auto_propagate_secs": round(total_auto_propagate, 6),
            "auto_masks_secs": round(total_auto_masks, 6),
            "other_auto_secs": round(total_other_auto, 6),
            "auto_total_secs": round(total_auto_time, 6),
            "manual_frames": total_manual_frames,
            "avg_manual_sec_per_frame": round(per_manual_frame_sec, 6),
            "fully_manual_est_sec": round(fully_manual_est_sec, 6),
            "time_saved_sec": round(savings_sec, 6),
            "share_manual_wall": round((total_manual_time/total_wall) if total_wall>0 else 0.0, 6),
            "share_propagation_wall": round((total_auto_propagate/total_wall) if total_wall>0 else 0.0, 6),
            "share_masks_wall": round((total_auto_masks/total_wall) if total_wall>0 else 0.0, 6),
            "share_other_wall": round((total_other_auto/total_wall) if total_wall>0 else 0.0, 6)
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
    ap = argparse.ArgumentParser(description="Timing & savings analysis for labeling logs.")
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
