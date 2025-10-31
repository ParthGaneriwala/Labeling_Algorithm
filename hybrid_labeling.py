"""
Hybrid Interactive and Automated Labeling Method

This module implements Algorithm 1: Hybrid Interactive and Automated Labeling Method
for lane marking annotation compatible with CLRerNet architecture.

The hybrid approach combines:
1. Manual point annotations on selected frames
2. Automated propagation across sequential frames using directional translation
3. Generation of segmentation masks from annotated points

This method addresses limitations in CDLEM and ALINA by reducing manual effort
while maintaining annotation consistency.
"""

import json
import os
from time import time
import cv2
import numpy as np
import re
import math

def _median(arr):
    a = sorted(arr)
    n = len(a)
    if n == 0:
        return 0.0
    if n % 2:
        return float(a[n // 2])
    return 0.5 * (a[n // 2 - 1] + a[n // 2])
def _read_lanes(file_path):
    """Load lanes from .lines.txt → list[list[(x,y),...]]; empty if missing."""
    lanes = []
    if not os.path.exists(file_path):
        return lanes
    with open(file_path, 'r') as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 4:
                continue
            pts = []
            for i in range(0, len(vals), 2):
                pts.append((int(round(float(vals[i]))),
                            int(round(float(vals[i+1])))))
            lanes.append(pts)
    return lanes

def _write_lanes_fresh(file_path, lanes):
    """Overwrite file with given lanes; one lane per line."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for lane in lanes:
            f.write(' '.join(f'{x} {y}' for x, y in lane) + '\n')
class HybridLabeler:
    """
    Implements the hybrid labeling algorithm for lane marking annotation.
    
    The algorithm processes image sequences by:
    - Manually annotating key frames at regular intervals (step size σ)
    - Automatically propagating annotations to intermediate frames
    - Generating binary segmentation masks for all frames
    """
    
    def __init__(self, step_size=60, adjustment_factor=1.5):
        """
        Initialize the hybrid labeler.
        
        Args:
            step_size (int): Number of frames between manual annotations (σ in Algorithm 1)
            adjustment_factor (float): Scaling factor for directional translation
        """
        self.step_size = step_size
        self.adjustment_factor = adjustment_factor
        self.current_points = []
        self.lanes_points = []
        self.lane_directions = []
        self.max_window_size = (1280, 800)  # (max_width, max_height)
        self.display_scale = 1.0
        self.drift_threshold_px = 8.0   # trigger if median drift > this (pixels)
        self.pause_ms = 50              # display delay for automated frames
        self.default_lane_mode = "single"   # or "multi"
        self.current_lane_mode = self.default_lane_mode
        self.zoom = 1.0
        self.pan = [0, 0]
        self.panning = False
        self.last_xy = (0,0)
        self.snap_to_edge = True   # toggle with 'e' during manual mode
        self.smooth_masks = False 



    

    def _to_orig(self, x, y):
        zx, zy = self.zoom, self.zoom
        px, py = self.pan
        return int(round((x - px) / zx)), int(round((y - py) / zy))
    def _fit_scale(self, image):
        """Compute a scale <= 1.0 so the image fits in max_window_size."""
        h, w = image.shape[:2]
        max_w, max_h = self.max_window_size
        scale = min(max_w / float(w), max_h / float(h), 1.0)
        return scale
    
    def _snap_to_edge(self, orig_img, x, y, pts, normal_search=24, roi_half=16,
                  canny1=50, canny2=150):
        """
        Refine (x,y) to the strongest edge near the local *normal* direction.
        - If we have at least 2 prior points, compute tangent from the last segment
        and search +/- normal; else fall back to best edge in a small ROI.
        Returns refined (xr, yr) in *original image* coords.
        """
        H, W = orig_img.shape[:2]
        x = int(np.clip(x, 0, W-1)); y = int(np.clip(y, 0, H-1))

        gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

        # ROI bounds
        x0 = max(0, x - roi_half); x1 = min(W, x + roi_half + 1)
        y0 = max(0, y - roi_half); y1 = min(H, y + roi_half + 1)
        roi = gray[y0:y1, x0:x1]
        edges = cv2.Canny(roi, canny1, canny2)

        # If we have a local tangent (>=2 points placed), search along normal
        if len(pts) >= 2:
            p2 = np.array(pts[-1], dtype=np.float32)
            p1 = np.array(pts[-2], dtype=np.float32)
            t = p2 - p1
            n = np.array([-t[1], t[0]], dtype=np.float32)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-6:
                n = np.array([0.0, 1.0], dtype=np.float32)
                n_norm = 1.0
            n /= n_norm

            best_val, best_xy = -1, (x, y)
            # sample along +/- normal, 1 pixel step
            for s in range(-normal_search, normal_search+1):
                xc = int(round(x + s * n[0]))
                yc = int(round(y + s * n[1]))
                if 0 <= xc < W and 0 <= yc < H:
                    # map to ROI coords
                    rx = xc - x0; ry = yc - y0
                    # local score: edges value + small 3x3 sum
                    if 0 <= rx < edges.shape[1] and 0 <= ry < edges.shape[0]:
                        v = int(edges[ry, rx])
                        # 3x3 neighborhood sum (clamped)
                        rx0 = max(0, rx-1); rx1 = min(edges.shape[1], rx+2)
                        ry0 = max(0, ry-1); ry1 = min(edges.shape[0], ry+2)
                        v += int(edges[ry0:ry1, rx0:rx1].sum()) // 9
                        if v > best_val:
                            best_val, best_xy = v, (xc, yc)
            return best_xy

        # Fallback: strongest edge in ROI
        ys, xs = np.where(edges > 0)
        if len(xs) > 0:
            # pick global max gradient magnitude around edge pixels
            sobx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
            soby = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(sobx**2 + soby**2)
            idx = np.argmax(mag[ys, xs])
            rx, ry = xs[idx], ys[idx]
            return (x0 + rx, y0 + ry)

        return (x, y)

    def _estimate_drift(self, prev_img, next_img, prev_points, predicted_points):
        """
        Compare motion from optical flow with our predicted translation.
        prev_points: list[(x,y)] in prev_img coords (original resolution)
        predicted_points: list[(x,y)] for next frame (from adjust_points)
        Returns median absolute error (pixels). Lower is better.
        """
        if len(prev_points) == 0:
            return 0.0

        # Prepare points for LK flow
        p0 = np.array(prev_points, dtype=np.float32).reshape(-1, 1, 2)
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, next_gray, p0, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # Keep only good tracks
        good_prev = p0[st.flatten() == 1].reshape(-1, 2)
        good_flow = p1[st.flatten() == 1].reshape(-1, 2) if p1 is not None else np.empty((0, 2))
        if good_prev.shape[0] == 0 or good_flow.shape[0] == 0:
            return 0.0

        # Our predicted targets at those same indices
        # Build a map from original prev_points order to predicted_points
        # (Assuming same ordering)
        predicted = np.array(predicted_points, dtype=np.float32).reshape(-1, 2)
        predicted_good = predicted[st.flatten() == 1]

        # Compute per-point error between flow result and predicted location
        diffs = np.linalg.norm(good_flow - predicted_good, axis=1)
        return _median(diffs.tolist())
       
    def gamma(self, direction):
        """
        Directional translation function γ(δ) from Algorithm 1.
        
        Maps lane direction to a translation vector for point adjustment.
        
        Args:
            direction (str): Lane direction ('left', 'right', 'up', 'down', 'straight')
            
        Returns:
            tuple: (dx, dy) translation vector
        """
        direction_map = {
            'left': (-self.adjustment_factor, 0),
            'right': (self.adjustment_factor, 0),
            'up': (0, -self.adjustment_factor),
            'down': (0, self.adjustment_factor),
            'straight': (0, 0)
        }
        return direction_map.get(direction, (0, 0))
    
    def adjust_points(self, points, k, i, direction):
        """
        Adjust points using directional translation function (Line 8 in Algorithm 1).
        
        Implements: P_k = P_i + (k - i) · γ(δ)
        
        Args:
            points (list): Initial lane points P_i
            k (int): Current frame index
            i (int): Initial frame index
            direction (str): Lane direction δ
            
        Returns:
            list: Adjusted points P_k
        """
        dx, dy = self.gamma(direction)
        frame_offset = k - i
        adjusted = [(x + frame_offset * dx, y + frame_offset * dy) for x, y in points]
        return adjusted
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        param: {'img': preview_img, 'scale': float, 'orig': original_image}
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            scale = param['scale']
            vis = param['img']
            orig = param.get('orig', None)

            # map preview -> original
            ox = int(round(x / scale))
            oy = int(round(y / scale))

            if self.snap_to_edge and orig is not None:
                ox, oy = self._snap_to_edge(orig, ox, oy, self.current_points)

            self.current_points.append((ox, oy))

            # draw on preview at preview coords (use the *clicked* preview position)
            cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)
            cv2.imshow("Image", vis)


    
    def manual_annotation(self, image, output_file):
        self.current_points = []
        direction = 'straight'
        self.current_lane_mode = self.default_lane_mode  # reset per frame
        lanes_tmp = []          # collect lanes here
        dirs_tmp = []           # parallel directions

        self.display_scale = self._fit_scale(image)
        img_vis = cv2.resize(image, None, fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_AREA)

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image",
                        min(int(image.shape[1] * self.display_scale), self.max_window_size[0]),
                        min(int(image.shape[0] * self.display_scale), self.max_window_size[1]))
        cv2.imshow("Image", img_vis)
        cv2.setMouseCallback("Image", self.mouse_callback,
                     {'img': img_vis, 'scale': self.display_scale, 'orig': image})


        print("Instructions:")
        print("  - Left click: Select lane points")
        print("  - 'k': Save current lane")
        print("  - '1' single-lane (only one lane will be kept)")
        print("  - '2' multi-lane (keep all)")
        print("  - Arrow keys: Set direction (Left/Right/Up/Down, default: Straight)")
        print("  - 'r': Redo current frame")
        print("  - Enter: Finish and write annotations")

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord('1'):
                self.current_lane_mode = "single"
                print("Lane mode: SINGLE")
            elif key == ord('2'):
                self.current_lane_mode = "multi"
                print("Lane mode: MULTI")

            elif key == ord('k'):
                if self.current_points:
                    if self.current_lane_mode == "single":
                        lanes_tmp = [self.current_points.copy()]
                        dirs_tmp  = [direction]
                    else:
                        lanes_tmp.append(self.current_points.copy())
                        dirs_tmp.append(direction)
                    self.current_points = []
                    direction = 'straight'
                    print(f"Saved lane. Total lanes (pending write): {len(lanes_tmp)}")

            elif key == 13:  # Enter
                if self.current_points:
                    if self.current_lane_mode == "single":
                        lanes_tmp = [self.current_points.copy()]
                        dirs_tmp  = [direction]
                    else:
                        lanes_tmp.append(self.current_points.copy())
                        dirs_tmp.append(direction)
                break

            elif key == ord('r'):
                print("Redoing current frame...")
                self.current_points = []
                lanes_tmp, dirs_tmp = [], []
                img_vis = cv2.resize(image, None, fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_AREA)
                cv2.imshow("Image", img_vis)
                cv2.setMouseCallback("Image", self.mouse_callback,
                     {'img': img_vis, 'scale': self.display_scale, 'orig': image})

            elif key == ord('e'):
                self.snap_to_edge = not self.snap_to_edge
                print(f"Snap-to-edge: {'ON' if self.snap_to_edge else 'OFF'}")

            elif key == 81:  # Left
                direction = 'left'; print(f"Direction: {direction}")
            elif key == 82:  # Up
                direction = 'up'; print(f"Direction: {direction}")
            elif key == 83:  # Right
                direction = 'right'; print(f"Direction: {direction}")
            elif key == 84:  # Down
                direction = 'down'; print(f"Direction: {direction}")
            elif key == ord('+'): 
                self.zoom = min(8.0, self.zoom*1.25)
            elif key == ord('-'): 
                self.zoom = max(0.1, self.zoom/1.25)

        cv2.destroyAllWindows()

        # Commit lanes officially to the class and write them FRESH
        # smooth + densify each lane before saving
        smoothed = []
        for lane in lanes_tmp:
            s = self._chaikin(lane, iters=2)
            s = self._resample(s, step=6)
            smoothed.append(s)
        self.lanes_points = smoothed

        self.lane_directions = dirs_tmp
        
        _write_lanes_fresh(output_file, self.lanes_points)


    
    def _save_points(self, file_path):
        """
        Save points to annotation file (Line 5 in Algorithm 1).
        
        Args:
            file_path (str): Path to annotation file
        """
        with open(file_path, 'a') as f:
            f.write(' '.join(f'{x} {y}' for x, y in self.current_points) + '\n')
    
    def _chaikin(self, pts, iters=2):
        """Chaikin corner-cutting that PRESERVES endpoints."""
        if len(pts) < 3:
            return pts
        P = [(float(x), float(y)) for x, y in pts]
        for _ in range(iters):
            Q = [P[0]]  # keep first endpoint
            for i in range(len(P) - 1):
                x0, y0 = P[i]
                x1, y1 = P[i + 1]
                Q.append((0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1))
                Q.append((0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1))
            Q.append(P[-1])  # keep last endpoint
            P = Q
        return [(int(round(x)), int(round(y))) for x, y in P]


    def _resample(self, pts, step=6):
        """Uniformly resample a polyline at ~step pixels spacing, INCLUDING the end."""
        if len(pts) < 2:
            return pts
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]

        # cumulative length
        d = [0.0]
        for i in range(1, len(pts)):
            dx = xs[i] - xs[i-1]; dy = ys[i] - ys[i-1]
            d.append(d[-1] + (dx*dx + dy*dy) ** 0.5)
        L = d[-1]
        if L <= 1e-6:
            return pts

        # include the end length L
        import numpy as np
        t = np.arange(0, L + 1e-6, step, dtype=np.float32)
        if t[-1] < L:
            t = np.append(t, L)

        x = np.interp(t, d, xs); y = np.interp(t, d, ys)
        # ensure exact last original point is present
        x[-1], y[-1] = xs[-1], ys[-1]
        return [(int(round(xx)), int(round(yy))) for xx, yy in zip(x, y)]

    
    @staticmethod
    def resample_polyline(pts, spacing=8):
        if len(pts) < 2: return pts
        # cumulative lengths
        d=[0.0]
        for i in range(1,len(pts)):
            dx=pts[i][0]-pts[i-1][0]; dy=pts[i][1]-pts[i-1][1]
            d.append(d[-1]+(dx*dx+dy*dy)**0.5)
        L=d[-1]; 
        if L==0: 
            return pts
        import numpy as np
        t = np.linspace(0,L,int(L/spacing)+1)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x = np.interp(t, d, xs); y = np.interp(t, d, ys)
        return list(map(lambda a:(int(round(a[0])),int(round(a[1]))), zip(x,y)))
    
    @staticmethod
    def lane_thickness(y, h, t_min=3, t_max=9):
        return int(round(t_min + (t_max - t_min) * (y / max(1,h-1))))

    
    @staticmethod
    def is_scene_cut(a, b, thresh=0.5):
        a_hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV); b_hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
        score = 0
        for ch in range(3):
            histA = cv2.calcHist([a_hsv],[ch],None,[32],[0,256])
            histB = cv2.calcHist([b_hsv],[ch],None,[32],[0,256])
            score += 1 - cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)
        return (score/3) > thresh

    
    @staticmethod
    def _write_ckpt(path, idx, file):
        import time
        with open(path, 'w') as f:
            json.dump({"last_index": idx, "file": file, "ts": time.time()}, f)

    @staticmethod
    def _write_atomic(path, text):
        tmp = path + ".tmp"
        with open(tmp, "w") as f: f.write(text)
        os.replace(tmp, path)

    def propagate_annotations(self, image, prev_image, prev_points_per_lane, i, k, output_file):
        """
        Propagate adjusted points to frame I_k, visualize, and optionally early-stop.
        prev_image: previous frame image (I_{k-1} or I_i)
        prev_points_per_lane: list of lists of points for previous frame (same structure as lanes_points)
        Returns: "ok" | "manual" | "abort", adjusted_points_per_lane
        """
        scale = self.display_scale if hasattr(self, 'display_scale') else self._fit_scale(image)
        img_vis = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        adjusted_points_per_lane = []   # anchor -> k (for save/draw)
        prev_concat = []                # prev-frame points (for LK)
        pred_concat = []                # prev points shifted by one step (for LK)

        # Guard if lane counts differ; pair by index
        num_lanes = min(len(self.lanes_points), len(prev_points_per_lane))

        for lane_idx in range(num_lanes):
            direction = self.lane_directions[lane_idx] if lane_idx < len(self.lane_directions) else 'straight'
            dx, dy = self.gamma(direction)

            # A) Anchor-based propagation (anchor i -> k)  — later smoothed for save/draw
            initial_points = self.lanes_points[lane_idx]
            adj_anchor = self.adjust_points(initial_points, k, i, direction)
            adjusted_points_per_lane.append(adj_anchor)

            # B) Prev->next prediction for drift (same cardinality as prev points)
            prev_pts = prev_points_per_lane[lane_idx]
            pred_prev = [(x + dx, y + dy) for (x, y) in prev_pts]

            prev_concat.extend(prev_pts)
            pred_concat.extend(pred_prev)

        # ---- Smooth/densify ONLY the anchor-based curves (for save/draw preview) ----
        for idx in range(len(adjusted_points_per_lane)):
            s = self._chaikin(adjusted_points_per_lane[idx], iters=2)
            s = self._resample(s, step=6)
            adjusted_points_per_lane[idx] = s

        # If single-lane mode, keep the longest propagated lane
        if self.current_lane_mode == "single" and len(adjusted_points_per_lane) > 1:
            adjusted_points_per_lane.sort(key=len, reverse=True)
            adjusted_points_per_lane = [adjusted_points_per_lane[0]]

        # Overwrite the .lines.txt once (fresh)
        _write_lanes_fresh(output_file, adjusted_points_per_lane)

        if not os.path.exists(output_file) or len(_read_lanes(output_file)) != len(adjusted_points_per_lane):
            print(f"Warning: mismatch writing {output_file}")

        # ---- Preview (draw smooth polylines) ----
        for lane in adjusted_points_per_lane:
            if len(lane) >= 2:
                pts = np.array([(int(x*scale), int(y*scale)) for x, y in lane], np.int32)
                cv2.polylines(img_vis, [pts], isClosed=False, color=(0,255,0), thickness=4, lineType=cv2.LINE_AA)

        # ---- DRIFT on matching lengths (prev_concat vs pred_concat) ----
        drift = self._estimate_drift(prev_image, image, prev_concat, pred_concat)

        # Adaptive threshold (tight when motion small, relaxed when motion large)
        base = float(self.drift_threshold_px)
        mag = 1.0
        if prev_points_per_lane and prev_points_per_lane[0] and len(prev_points_per_lane[0]) >= 2:
            arr = np.array(prev_points_per_lane[0], dtype=np.float32)
            diffs = np.diff(arr, axis=0)
            step_mags = np.linalg.norm(diffs, axis=1)
            if step_mags.size:
                mag = float(np.median(step_mags))
        thresh = float(np.clip(base * (1.0 + 0.5 * (mag / 3.0)), 4.0, 20.0))
        self.last_drift_threshold = thresh

        if drift > thresh:
            cv2.putText(img_vis,
                        f"DRIFT {drift:.1f}px  thr~{thresh:.1f}  [m]=manual  [c]=continue  [q]=abort",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        # Show window then wait using the ADAPTIVE threshold
        cv2.namedWindow("Automated Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Automated Frame",
                        min(int(image.shape[1] * scale), self.max_window_size[0]),
                        min(int(image.shape[0] * scale), self.max_window_size[1]))
        cv2.imshow("Automated Frame", img_vis)

        wait = 1000 if drift > thresh else self.pause_ms
        key = cv2.waitKey(wait) & 0xFF

        if key in (ord('q'), 27):  # q or ESC
            return "abort", adjusted_points_per_lane
        if key == ord('m'):        # switch to manual on this frame
            return "manual", adjusted_points_per_lane
        return "ok", adjusted_points_per_lane


    
    def process_image_sequence(self, input_dir, output_dir, resume=True):
        os.makedirs(output_dir, exist_ok=True)
        files = sorted(
            [f for f in os.listdir(input_dir) if f.endswith('.jpg')],
            key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf')
        )
        N = len(files)
        if N == 0:
            print("No images found.")
            return

        # --- Find resume start ---
        start_idx = 0
        if resume:
            for idx, fname in enumerate(files):
                if not os.path.exists(os.path.join(output_dir, fname.replace('.jpg', '.lines.txt'))):
                    start_idx = idx
                    break
            else:
                print("All frames already annotated. Nothing to do.")
                return

        i = start_idx
        while i < N:
            image_path = os.path.join(input_dir, files[i])
            image = cv2.imread(image_path)
            out_i = os.path.join(output_dir, files[i].replace('.jpg', '.lines.txt'))

            print(f"Processing anchor frame {i+1}/{N} - {files[i]}")

            # If this anchor already has annotations and we're resuming, load them; else collect manually
            existing = _read_lanes(out_i) if resume else []
            if existing:
                print("Existing annotations found for anchor; using them.")
                self.lanes_points = existing
                # no stored directions on disk; default to 'straight'
                self.lane_directions = ['straight'] * len(existing)
            else:
                self.lanes_points = []
                self.lane_directions = []
                self.manual_annotation(image, out_i)  # writes fresh

            # Prepare prev refs from anchor
            prev_image = image
            prev_points_per_lane = [lp.copy() for lp in self.lanes_points]

            jumped_to_manual = False
            for j in range(1, self.step_size):
                k = i + j
                if k >= N:
                    break

                next_image_path = os.path.join(input_dir, files[k])
                next_image = cv2.imread(next_image_path)
                out_k = os.path.join(output_dir, files[k].replace('.jpg', '.lines.txt'))

                # If annotation already exists for k, load and continue (resume mid-run)
                existing_k = _read_lanes(out_k) if resume else []
                if existing_k:
                    prev_image = next_image
                    prev_points_per_lane = [lp.copy() for lp in existing_k]
                    continue

                # Otherwise, propagate now
                scale = self.display_scale if hasattr(self, 'display_scale') else self._fit_scale(next_image)
                img_vis = cv2.resize(next_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                status, adjusted_points_per_lane = self.propagate_annotations(
                    next_image, prev_image, prev_points_per_lane, i, k, out_k  # out_k is written inside
                )

                if status == "abort":
                    cv2.destroyAllWindows()
                    print("Aborted by user.")
                    return

                if status == "manual":
                    print(f"Switching to manual at frame {k+1}/{N} - {files[k]}")
                    self.lanes_points = []
                    self.lane_directions = []
                    self.manual_annotation(next_image, out_k)  # writes fresh
                    # Restart from here
                    i = k
                    jumped_to_manual = True
                    break


                prev_image = next_image
                prev_points_per_lane = [lp.copy() for lp in adjusted_points_per_lane]

            cv2.destroyAllWindows()

            if not jumped_to_manual:
                i += self.step_size
    
    def generate_segmentation_masks(self, base_dir, annotations_dir, output_dir, image_list_file=None):
        """
        Generate segmentation masks M_i from lane points (Lines 14-19 in Algorithm 1).
        
        Implements mask generation: M_i(x,y) = 1 if (x,y) in lane segment, 0 otherwise
        
        Args:
            base_dir (str): Base directory containing images
            annotations_dir (str): Directory with .lines.txt annotation files
            output_dir (str): Directory to save segmentation masks
            image_list_file (str): Optional file listing images to process
        """
        mask_dir = os.path.join(output_dir, 'images')
        os.makedirs(mask_dir, exist_ok=True)
        
        # Get list of images to process
        if image_list_file and os.path.exists(image_list_file):
            with open(image_list_file, 'r') as f:
                image_paths = [line.strip() for line in f.readlines()]
        else:
            # Process all images in annotations directory
            image_paths = [f.replace('.lines.txt', '.jpg') 
                          for f in os.listdir(annotations_dir) 
                          if f.endswith('.lines.txt')]
        
        output_list_file = os.path.join(output_dir, 'segmentation_list.txt')
        
        with open(output_list_file, 'w') as out_list:
            for img_path in image_paths:
                if image_list_file:
                    full_image_path = os.path.join(base_dir, img_path)
                else:
                    full_image_path = os.path.join(base_dir, img_path)
                
                annotation_file = os.path.join(annotations_dir, 
                                              os.path.basename(img_path).replace('.jpg', '.lines.txt'))
                
                # Load image
                image = cv2.imread(full_image_path)
                if image is None:
                    print(f"Warning: Could not load image {full_image_path}")
                    continue
                
                # Create single-channel mask once per image
                h, w = image.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)

                if os.path.exists(annotation_file):
                    with open(annotation_file, 'r') as f:
                        for lane_idx, line in enumerate(f):
                            points = line.strip().split()
                            if len(points) < 4:
                                continue

                            # Extract + optional smooth/resample for cleaner masks
                            point_coords = []
                            for i_pt in range(0, len(points), 2):
                                x = int(round(float(points[i_pt])))
                                y = int(round(float(points[i_pt+1])))
                                point_coords.append((x, y))
                            if self.smooth_masks:
                                point_coords = self._chaikin(point_coords, iters=2)
                                point_coords = self._resample(point_coords, step=4)

                            # Draw lane into the SAME mask
                            for j in range(len(point_coords) - 1):
                                pt1, pt2 = point_coords[j], point_coords[j+1]
                                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                                    cv2.line(mask, pt1, pt2, 255, 5, lineType=cv2.LINE_AA)

                # Save mask (Line 19)
                mask_filename = os.path.join(mask_dir, 
                                            os.path.basename(full_image_path).replace('.jpg', '.png'))
                cv2.imwrite(mask_filename, mask)
                
                # Write to output list
                if image_list_file:
                    out_list.write(f"{img_path} {os.path.relpath(mask_filename, base_dir)}\n")
                else:
                    out_list.write(f"{os.path.basename(img_path)} {os.path.basename(mask_filename)}\n")
        
        print(f"Segmentation masks saved to {mask_dir}")
        print(f"Output list saved to {output_list_file}")

def main():
    """
    Example usage of the hybrid labeling algorithm.
    """
    # Configuration
    input_dir = 'E:\\ExtractedFrames\\GX010009'
    output_dir = 'E:\\ExtractedFrames\\GX010009\\annotations'
    step_size = 60  # σ in Algorithm 1
    adjustment_factor = 1.5  # Scaling for γ(δ)
    
    # Initialize labeler
    labeler = HybridLabeler(step_size=step_size, adjustment_factor=adjustment_factor)
    
    # Phase 1: Process image sequence with hybrid labeling (Lines 1-13)
    print("=" * 60)
    print("Phase 1: Hybrid Interactive and Automated Labeling")
    print("=" * 60)
    labeler.process_image_sequence(input_dir, output_dir)
    
    # Phase 2: Generate segmentation masks (Lines 14-19)
    print("\n" + "=" * 60)
    print("Phase 2: Segmentation Mask Generation")
    print("=" * 60)
    mask_output_dir = 'E:\\ExtractedFrames\\GX010009\\segmentation_masks'
    labeler.generate_segmentation_masks(
        base_dir='E:\\ExtractedFrames\\GX010009',
        annotations_dir=output_dir,
        output_dir=mask_output_dir
    )
    
    print("\n" + "=" * 60)
    print("Hybrid labeling completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
