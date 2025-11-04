#!/usr/bin/env python3
# dataset_info.py
import argparse, csv, math
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

# ---------- IO ----------
def read_all_lanes(lines_path: Path) -> List[List[Tuple[int,int]]]:
    lanes = []
    if not lines_path.exists(): return lanes
    with open(lines_path, "r") as f:
        for raw in f:
            vals = raw.strip().split()
            if len(vals) < 4:
                continue
            pts=[]; ok=True
            for i in range(0, len(vals), 2):
                try:
                    x = int(round(float(vals[i]))); y = int(round(float(vals[i+1])))
                except Exception:
                    ok=False; break
                pts.append((x,y))
            if ok and len(pts)>=2:
                lanes.append(pts)
    return lanes

# ---------- Geometry helpers ----------
def resample_polyline(pts: List[Tuple[float,float]], step: float=8.0) -> List[Tuple[float,float]]:
    if len(pts) < 2: return pts
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    d=[0.0]
    for i in range(1, len(pts)):
        dx=xs[i]-xs[i-1]; dy=ys[i]-ys[i-1]
        d.append(d[-1] + math.hypot(dx,dy))
    L=d[-1]
    if L<=1e-6: return pts
    t=np.arange(0.0, L+1e-6, step, dtype=np.float32)
    if t[-1] < L: t=np.append(t, L)
    x=np.interp(t,d,xs); y=np.interp(t,d,ys)
    x[-1], y[-1] = xs[-1], ys[-1]
    return list(zip(x.tolist(), y.tolist()))

def arc_length(pts): 
    if len(pts)<2: return 0.0
    return sum(math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]) for i in range(1,len(pts)))

def heading_angles_deg(pts):
    hs=[]
    for i in range(1,len(pts)):
        dx = pts[i][0]-pts[i-1][0]
        dy = pts[i][1]-pts[i-1][1]
        if dx==0 and dy==0: continue
        hs.append(math.degrees(math.atan2(dy, dx)))
    return hs

def circular_std_deg(angles):
    """Circular standard deviation (degrees), robust to float error."""
    if not angles:
        return 0.0
    a = np.radians(np.asarray(angles, dtype=float))
    C = float(np.mean(np.cos(a)))
    S = float(np.mean(np.sin(a)))
    R = math.hypot(C, S)
    eps = 1e-12
    R = min(max(R, eps), 1.0 - eps)
    return math.degrees(math.sqrt(max(0.0, -2.0 * math.log(R))))

def pca_axis(pts):
    A=np.array(pts, dtype=np.float64)
    mu=A.mean(axis=0, keepdims=True)
    X=A-mu
    if X.shape[0]<2:
        return mu[0], np.array([1.0,0.0]), 0.0
    C=np.cov(X.T)
    evals, evecs=np.linalg.eigh(C)
    idx=np.argsort(evals)[::-1]
    evals=evals[idx]; evecs=evecs[:,idx]
    u=evecs[:,0]  # principal direction (unit)
    lam1=max(evals[0],1e-12); lam2=max(evals[1],0.0)
    eig_ratio=lam2/lam1
    return mu[0], u, eig_ratio

def total_turn_deg(pts):
    if len(pts)<3: return 0.0
    s=0.0
    for i in range(1,len(pts)-1):
        a=np.array(pts[i]) - np.array(pts[i-1])
        b=np.array(pts[i+1]) - np.array(pts[i])
        na,nb=np.linalg.norm(a),np.linalg.norm(b)
        if na<1e-9 or nb<1e-9: 
            continue
        cosang=np.clip(np.dot(a,b)/(na*nb), -1.0, 1.0)
        s += math.degrees(math.acos(cosang))
    return s

def quantile_perp_dev_pixels(pts, trim_frac=0.10, q=0.95):
    """Quantile of perpendicular deviation (pixels) after trimming ends by arc-length."""
    mu, u, _ = pca_axis(pts)
    A = np.array(pts, dtype=np.float64)
    X = A - mu
    dot = X @ u
    sqnorm = np.sum(X*X, axis=1)
    perp = np.sqrt(np.maximum(sqnorm - dot*dot, 0.0))

    # arc-length along the polyline to trim ends
    xs = A[:,0]; ys = A[:,1]
    d = [0.0]
    for i in range(1, len(A)):
        d.append(d[-1] + math.hypot(xs[i]-xs[i-1], ys[i]-ys[i-1]))
    L = d[-1] if d else 0.0
    if L <= 1e-6 or len(perp) == 0:
        return float(np.max(perp) if len(perp) else 0.0)

    lo = trim_frac * L
    hi = (1.0 - trim_frac) * L
    keep = [(lo <= d[i] <= hi) for i in range(len(d))]
    sel = perp[keep] if np.any(keep) else perp
    return float(np.quantile(sel, q))

# ---------- Classifiers ----------
def classify_global(pts, resample_step, tau_eig, tau_turn_per100, tau_dev_over_len, min_len):
    pts = [(float(x), float(y)) for x,y in pts]
    if resample_step>0:
        pts = resample_polyline(pts, step=resample_step)
    L = arc_length(pts)
    if L < min_len:
        return "skipped", {"reason":"too_short", "length_px": L}
    _, _, eig_ratio = pca_axis(pts)
    turn_per100 = (total_turn_deg(pts)/L)*100.0 if L>0 else 0.0
    mu, u, _ = pca_axis(pts)
    X = np.array(pts, dtype=np.float64) - mu
    dot = X @ u
    sqnorm = np.sum(X*X, axis=1)
    perp = np.sqrt(np.maximum(sqnorm - dot*dot, 0.0))
    dev_over_len = float(np.max(perp) / max(L,1e-6))
    straight = (eig_ratio <= tau_eig) and (turn_per100 <= tau_turn_per100) and (dev_over_len <= tau_dev_over_len)
    return ("straight" if straight else "curved"), {
        "length_px": L, "eig_ratio": eig_ratio,
        "turn_deg_per_100px": turn_per100,
        "max_perp_dev_over_len": dev_over_len
    }

def classify_pixel(pts, resample_step, pix_dev, head_std_deg_max, min_len,
                   trim_ends=0.10, dev_quantile=0.95):
    pts = [(float(x), float(y)) for x,y in pts]
    if resample_step>0:
        pts = resample_polyline(pts, step=resample_step)
    L = arc_length(pts)
    if L < min_len:
        return "skipped", {"reason":"too_short", "length_px": L}
    dev_px = quantile_perp_dev_pixels(pts, trim_frac=trim_ends, q=dev_quantile)
    head_std = circular_std_deg(heading_angles_deg(pts))
    straight = (dev_px <= pix_dev) and (head_std <= head_std_deg_max)
    return ("straight" if straight else "curved"), {
        "length_px": L, "max_perp_dev_px_q": dev_px,
        "heading_std_deg": head_std, "trim_ends": trim_ends, "dev_quantile": dev_quantile
    }

def classify_window(pts, resample_step, win, win_step, win_turn, win_head_std, win_frac, min_len):
    pts = [(float(x), float(y)) for x,y in pts]
    if resample_step>0:
        pts = resample_polyline(pts, step=resample_step)
    L = arc_length(pts)
    if L < min_len:
        return "skipped", {"reason":"too_short", "length_px": L}
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    d=[0.0]
    for i in range(1,len(pts)):
        d.append(d[-1] + math.hypot(xs[i]-xs[i-1], ys[i]-ys[i-1]))
    straight_windows = 0; total_windows = 0
    if d[-1] < win:
        hs = heading_angles_deg(pts)
        turn = total_turn_deg(pts)
        head_std = circular_std_deg(hs)
        if (turn/ max(L,1e-6) * 100.0 <= win_turn) and (head_std <= win_head_std):
            straight_windows = 1
        total_windows = 1
    else:
        import bisect
        start = 0.0
        while start + win <= d[-1] + 1e-6:
            end = start + win
            i0 = bisect.bisect_left(d, start)
            i1 = bisect.bisect_left(d, end)
            seg = pts[max(0,i0-1):min(len(pts), i1+1)]
            if len(seg) >= 2:
                hs = heading_angles_deg(seg)
                turn = total_turn_deg(seg)
                head_std = circular_std_deg(hs)
                seg_len = max(arc_length(seg), 1e-6)
                if (turn/ seg_len * 100.0 <= win_turn) and (head_std <= win_head_std):
                    straight_windows += 1
                total_windows += 1
            start += win_step
        if total_windows == 0:
            total_windows = 1
    frac = straight_windows / total_windows
    straight = frac >= win_frac
    return ("straight" if straight else "curved"), {
        "length_px": L, "straight_window_frac": frac, "windows": total_windows
    }

# ---------- Walk & aggregate ----------
def main():
    ap = argparse.ArgumentParser(description="Dataset info: straight vs curved lanes (multi-lane per file).")
    ap.add_argument("annotations_dir", help="Folder with *.lines.txt")
    ap.add_argument("--method", choices=["global","pixel","window"], default="pixel",
                    help="Classifier to use.")
    ap.add_argument("--resample", type=float, default=8.0, help="Resample spacing (px).")

    # global (legacy) thresholds
    ap.add_argument("--tau-eig", "--tau_eig", dest="tau_eig", type=float, default=0.05)
    ap.add_argument("--tau-turn", "--tau_turn", dest="tau_turn", type=float, default=20.0)
    ap.add_argument("--tau-dev", "--tau_dev", dest="tau_dev", type=float, default=0.08)

    # pixel classifier thresholds
    ap.add_argument("--pix-dev", "--pix_dev", dest="pix_dev", type=float, default=6.0,
                    help="Max perpendicular deviation (pixels).")
    ap.add_argument("--head-std", "--head_std", dest="head_std", type=float, default=10.0,
                    help="Max circular stdev of segment headings (deg).")
    ap.add_argument("--dev-quantile", "--dev_quantile", dest="dev_quantile",
                    type=float, default=0.95, help="Quantile of deviation to use (0..1).")
    ap.add_argument("--trim-ends", "--trim_ends", dest="trim_ends",
                    type=float, default=0.10, help="Trim fraction from each end by arc length (0..0.4).")

    # windowed classifier thresholds
    ap.add_argument("--win", type=float, default=120.0, help="Window arc length (px).")
    ap.add_argument("--win-step", "--win_step", dest="win_step", type=float, default=60.0)
    ap.add_argument("--win-turn", "--win_turn", dest="win_turn", type=float, default=8.0,
                    help="Max turning deg per 100px inside a window.")
    ap.add_argument("--win-head-std", "--win_head_std", dest="win_head_std", type=float, default=6.0)
    ap.add_argument("--win-straight-frac", "--win_straight_frac", dest="win_straight_frac",
                    type=float, default=0.75, help="Fraction of windows that must be straight.")

    ap.add_argument("--min-len", "--min_len", dest="min_len", type=float, default=30.0)
    ap.add_argument("--perlane-csv", "--perlane_csv", dest="perlane_csv", type=str, default=None)
    ap.add_argument("--perimage-csv", "--perimage_csv", dest="perimage_csv", type=str, default=None)
    args = ap.parse_args()

    def classify(pts):
        if args.method == "global":
            return classify_global(pts, args.resample, args.tau_eig, args.tau_turn, args.tau_dev, args.min_len)
        elif args.method == "pixel":
            return classify_pixel(pts, args.resample, args.pix_dev, args.head_std, args.min_len,
                                  trim_ends=args.trim_ends, dev_quantile=args.dev_quantile)
        else:
            return classify_window(pts, args.resample, args.win, args.win_step, args.win_turn,
                                   args.win_head_std, args.win_straight_frac, args.min_len)

    root = Path(args.annotations_dir)
    files = sorted(root.rglob("*.lines.txt")) if root.is_dir() else ([root] if root.suffix==".txt" else [])
    if not files:
        raise SystemExit(f"No *.lines.txt found under {root}")

    total_lanes = straight_lanes = curved_lanes = skipped_lanes = 0
    perlane_rows: List[Dict] = []
    perimage_rows: List[Dict] = []

    for f in files:
        lanes = read_all_lanes(f)
        n_s = n_c = n_k = 0
        for li, pts in enumerate(lanes):
            label, m = classify(pts)
            total_lanes += 1
            if label == "straight":
                straight_lanes += 1; n_s += 1
            elif label == "curved":
                curved_lanes += 1; n_c += 1
            else:
                skipped_lanes += 1; n_k += 1
            if args.perlane_csv:
                row = {"file": str(f), "lane_index": li, "label": label}
                for k,v in m.items(): row[k]=v
                perlane_rows.append(row)

        perimage_rows.append({
            "file": str(f),
            "straight_lanes": n_s,
            "curved_lanes": n_c,
            "skipped_lanes": n_k,
            "total_lanes": n_s+n_c+n_k
        })

    print("==== Dataset lane-shape summary (per-lane counting) ====")
    print(f"Annotation files:     {len(files)}")
    print(f"Lanes total:          {total_lanes}")
    print(f"  Straight lanes:     {straight_lanes}")
    print(f"  Curved lanes:       {curved_lanes}")
    print(f"  Skipped lanes:      {skipped_lanes}")
    if total_lanes:
        ps = 100.0*straight_lanes/total_lanes
        pc = 100.0*curved_lanes/total_lanes
        print(f"Percent straight:     {ps:.1f}%")
        print(f"Percent curved:       {pc:.1f}%")

    if args.perlane_csv and perlane_rows:
        with open(args.perlane_csv, "w", newline="", encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(perlane_rows[0].keys()))
            w.writeheader(); w.writerows(perlane_rows)
        print(f"Wrote per-lane metrics: {args.perlane_csv}")

    if args.perimage_csv and perimage_rows:
        with open(args.perimage_csv, "w", newline="", encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(perimage_rows[0].keys()))
            w.writeheader(); w.writerows(perimage_rows)
        print(f"Wrote per-image lane counts: {args.perimage_csv}")

if __name__ == "__main__":
    import numpy as np
    main()
