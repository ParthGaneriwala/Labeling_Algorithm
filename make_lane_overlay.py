#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create lane overlay/mask from point annotations written for 3840x2160.
- Supports any input image size by scaling points from reference size.
- Outputs: overlay.png, mask.png (binary), blended.png
- Optional: a .pptx slide if python-pptx is installed.
"""

import os, argparse
import cv2
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser(description="Lane overlay/mask generator")
    ap.add_argument("--image", required=True, help="Path to input image (JPG/PNG, any size)")
    ap.add_argument("--points", required=True, help="Path to .lines.txt (one lane per line: x1 y1 x2 y2 ...)")
    ap.add_argument("--outdir", default="lane_outputs", help="Output directory")
    ap.add_argument("--refw", type=int, default=3840, help="Reference width of annotations (default: 3840)")
    ap.add_argument("--refh", type=int, default=2160, help="Reference height of annotations (default: 2160)")
    ap.add_argument("--thickness", type=int, default=8, help="Line thickness for drawing")
    ap.add_argument("--alpha", type=float, default=0.35, help="Blending alpha for preview")
    ap.add_argument("--pptx", action="store_true", help="Also generate a PPTX slide (requires python-pptx)")
    return ap.parse_args()

def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def read_lanes(points_path):
    """Return list[list[(x,y),...]] from .lines.txt"""
    lanes = []
    with open(points_path, "r") as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 4:
                continue
            pts = []
            for i in range(0, len(vals), 2):
                x = float(vals[i]); y = float(vals[i+1])
                pts.append((x, y))
            lanes.append(pts)
    if not lanes:
        print(f"Warning: no valid lanes found in {points_path}")
    return lanes

def scale_lanes(lanes, ref_w, ref_h, dst_w, dst_h):
    sx = dst_w / float(ref_w)
    sy = dst_h / float(ref_h)
    scaled = []
    for lane in lanes:
        scaled.append([(int(round(x*sx)), int(round(y*sy))) for (x,y) in lane])
    return scaled, sx, sy

def draw_overlay_and_mask(image, lanes_xy, thickness=8, aa=True):
    h, w = image.shape[:2]
    overlay = image.copy()
    mask = np.zeros((h, w), dtype=np.uint8)  # single-channel binary
    lt = cv2.LINE_AA if aa else cv2.LINE_8
    for pts in lanes_xy:
        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i+1]
            # bounds check just in case
            if (0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h):
                cv2.line(overlay, p1, p2, (0, 255, 0), thickness, lineType=lt)
                cv2.line(mask,   p1, p2, 255,          thickness, lineType=lt)
    return overlay, mask

def save_images(outdir, base_name, image, overlay, mask, alpha):
    os.makedirs(outdir, exist_ok=True)
    blended = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
    overlay_path = os.path.join(outdir, f"{base_name}_overlay.png")
    mask_path    = os.path.join(outdir, f"{base_name}_mask.png")
    blended_path = os.path.join(outdir, f"{base_name}_blended.png")
    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(blended_path, blended)
    return overlay_path, mask_path, blended_path

def main():
    args = parse_args()
    img = read_image(args.image)
    lanes = read_lanes(args.points)
    h, w = img.shape[:2]

    # scale annotation points from 3840x2160 to current image size
    lanes_xy, sx, sy = scale_lanes(lanes, args.refw, args.refh, w, h)

    overlay, mask = draw_overlay_and_mask(img, lanes_xy, thickness=args.thickness)
    base = os.path.splitext(os.path.basename(args.image))[0]
    overlay_path, mask_path, blended_path = save_images(args.outdir, base, img, overlay, mask, args.alpha)

    print("Saved:")
    print("  Overlay:", overlay_path)
    print("  Mask   :", mask_path)
    print("  Blended:", blended_path)


if __name__ == "__main__":
    main()
