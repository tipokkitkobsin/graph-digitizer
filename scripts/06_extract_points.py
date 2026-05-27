"""
Phase 5 — multi-series data extraction + CSV export.

Inside the YOLO-detected plot bbox:
  1. Find each distinct data-series COLOR by clustering high-saturation pixels in hue space.
  2. For each color, build a binary mask and run the chart-type-specific extractor:
        scatter -> connected components -> blob centroid + simple marker-shape descriptor
        line    -> skeletonize + median y per x column
        bar     -> column projection -> top of each bar
  3. Filter likely DISTRACTORS (axhline/axvline/trend lines) using shape heuristics:
       * long, thin, near-horizontal/vertical blobs that span >70 % of the plot width/height
       * blobs with very low saturation (already excluded by sat-mask)
  4. Map (pixel_x, pixel_y) -> (data_x, data_y) using Phase 4 calibration.
  5. Write one CSV per image with columns:
        series, color_hex, marker_guess, x, y, pixel_x, pixel_y
  6. Optional: per-series Hungarian match against ground truth (--meta-dir).

Inputs : Phase 4 JSON (--phase4-json), Phase 4 plot-bbox + axis calibration
Outputs: <out-dir>/<image_stem>.csv per image, plus summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from skimage.morphology import skeletonize


HUE_BINS = 180  # OpenCV HSV hue range is 0..179
MIN_PIXELS_FOR_SERIES = 25
MAX_SERIES = 5
HUE_PEAK_MIN_DIST = 8       # min hue distance between distinct series colors
MIN_PEAK_FRACTION = 0.07    # peak must be >= this fraction of max bin

DISTRACTOR_ASPECT_FRAC = 0.70   # blob spans >70% of plot width/height -> likely distractor
DISTRACTOR_MIN_AR = 8.0         # aspect ratio (long/thin) -> likely line


# ---------------- helpers ----------------

def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def bgr_to_hex(bgr) -> str:
    b, g, r = [int(v) for v in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def pixel_to_data(px, py, x_calib, y_calib):
    return (float(x_calib["slope"] * px + x_calib["intercept"]),
            float(y_calib["slope"] * py + y_calib["intercept"]))


def crop_plot(img, xyxy):
    x0, y0, x1, y1 = [int(round(v)) for v in xyxy]
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(img.shape[1], x1); y1 = min(img.shape[0], y1)
    return img[y0:y1, x0:x1].copy(), (x0, y0)


# ---------------- color clustering ----------------

def find_series_hues(plot_bgr, max_series=MAX_SERIES) -> list[int]:
    """Return list of dominant hue centers (OpenCV scale 0..179) for data series."""
    hsv = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2HSV)
    sat_mask = hsv[..., 1] > 80
    val_mask = hsv[..., 2] < 254
    valid = sat_mask & val_mask
    hues = hsv[..., 0][valid]
    if hues.size < MIN_PIXELS_FOR_SERIES:
        return []
    hist, _ = np.histogram(hues, bins=HUE_BINS, range=(0, HUE_BINS))
    # Pad cyclically for hue wrap-around handling
    hist = hist.astype(float)
    smooth = gaussian_filter1d(hist, sigma=2.0, mode="wrap")
    peaks, _ = find_peaks(smooth,
                          height=smooth.max() * MIN_PEAK_FRACTION,
                          distance=HUE_PEAK_MIN_DIST)
    order = np.argsort(smooth[peaks])[::-1]
    peaks = peaks[order][:max_series]
    return [int(p) for p in peaks]


def hue_mask(plot_bgr, hue_center: int, half_window: int = 7):
    """Pixels within +-half_window of hue_center on the cyclic hue axis."""
    hsv = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(int)
    sat_mask = hsv[..., 1] > 80
    val_mask = hsv[..., 2] < 254
    d = np.minimum((h - hue_center) % HUE_BINS, (hue_center - h) % HUE_BINS)
    return (d <= half_window) & sat_mask & val_mask


def median_bgr_in_mask(plot_bgr, mask) -> tuple[int, int, int]:
    if not mask.any():
        return (0, 0, 0)
    pixels = plot_bgr[mask]
    return tuple(int(v) for v in np.median(pixels, axis=0))


# ---------------- per-color extractors ----------------

def is_distractor_blob(stats_row, mask_shape) -> bool:
    """Heuristic: long, thin blob spanning most of the plot is a reference/trend line."""
    w = int(stats_row[cv2.CC_STAT_WIDTH])
    h = int(stats_row[cv2.CC_STAT_HEIGHT])
    plot_h, plot_w = mask_shape
    if w == 0 or h == 0:
        return False
    aspect = max(w / max(1, h), h / max(1, w))
    spans_width = w >= DISTRACTOR_ASPECT_FRAC * plot_w
    spans_height = h >= DISTRACTOR_ASPECT_FRAC * plot_h
    return aspect >= DISTRACTOR_MIN_AR and (spans_width or spans_height)


def extract_scatter_from_mask(mask, origin_xy, plot_shape):
    ox, oy = origin_xy
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    points = []
    markers = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (8 <= area <= 2500):
            continue
        if is_distractor_blob(stats[i], plot_shape):
            continue
        cx, cy = centroids[i]
        # Crude marker shape: aspect ratio + circularity
        comp_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(cnt, True)
            circ = 4 * np.pi * area / max(peri ** 2, 1.0)
            marker = "round" if circ > 0.65 else "angular"
        else:
            marker = "unknown"
        points.append((cx + ox, cy + oy))
        markers.append(marker)
    return points, markers


def extract_line_from_mask(mask, origin_xy, plot_shape):
    ox, oy = origin_xy
    if mask.sum() < 30:
        return [], []
    # Filter giant distractor blobs FIRST (a single trend line would
    # have nearly the same color as a series, but typically thinner)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask, dtype=bool)
    for i in range(1, n_labels):
        if is_distractor_blob(stats[i], plot_shape):
            continue
        if stats[i, cv2.CC_STAT_AREA] < 15:
            continue
        keep |= (labels == i)
    if keep.sum() < 30:
        return [], []
    skel = skeletonize(keep)
    h, w = skel.shape
    points = []
    for x in range(w):
        ys = np.where(skel[:, x])[0]
        if ys.size:
            y = float(np.median(ys))
            points.append((x + ox, y + oy))
    if len(points) > 80:
        step = max(1, len(points) // 80)
        points = points[::step]
    return points, [None] * len(points)


def extract_bar_from_mask(mask, origin_xy, plot_shape):
    ox, oy = origin_xy
    mask = mask.astype(np.uint8)
    if mask.sum() < 30:
        return [], []
    plot_h, _ = plot_shape
    col_counts = mask.sum(axis=0)
    in_bar = col_counts > (plot_h * 0.05)
    points = []
    i = 0
    n = len(in_bar)
    while i < n:
        if in_bar[i]:
            j = i
            while j < n and in_bar[j]:
                j += 1
            bar_cols = mask[:, i:j]
            row_any = bar_cols.any(axis=1)
            top_rows = np.where(row_any)[0]
            if top_rows.size:
                top_y = float(top_rows.min())
                cx = (i + j) / 2.0
                points.append((cx + ox, top_y + oy))
            i = j
        else:
            i += 1
    return points, [None] * len(points)


def extract_scatter_with_line_filter(mask, origin_xy, plot_shape):
    """Like extract_scatter_from_mask, but stricter on circularity + area so the
    overlaid fit line (typically one elongated, large component per color) is
    excluded. Used for the `line_with_scatter` class where each color contains
    both raw scatter points AND a smooth fit curve."""
    ox, oy = origin_xy
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    points, markers = [], []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        # Tighter area band (line components are usually >> 1200)
        if not (8 <= area <= 1200):
            continue
        if is_distractor_blob(stats[i], plot_shape):
            continue
        comp_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        circ = 4 * np.pi * area / max(peri ** 2, 1.0)
        # Hard reject elongated shapes — that's the fit line
        if circ < 0.5:
            continue
        cx, cy = centroids[i]
        points.append((cx + ox, cy + oy))
        markers.append("round" if circ > 0.65 else "angular")
    return points, markers


EXTRACTORS = {
    "scatter_plot": extract_scatter_from_mask,
    "line_plot": extract_line_from_mask,
    "bar_plot": extract_bar_from_mask,
    "line_with_scatter": extract_scatter_with_line_filter,
}


# ---------------- ground-truth comparison ----------------

def match_series_by_color(extracted_series, gt_series):
    """Greedy match each extracted series to the GT series with the closest color (LAB distance)."""
    if not extracted_series or not gt_series:
        return {}
    ex_colors_lab = []
    for s in extracted_series:
        bgr = np.uint8([[hex_to_rgb(s["color_hex"])[::-1]]])  # RGB->BGR for cvtColor
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0]
        ex_colors_lab.append(lab.astype(float))
    gt_colors_lab = []
    for s in gt_series:
        rgb = hex_to_rgb(s["color"])
        bgr = np.uint8([[(rgb[2], rgb[1], rgb[0])]])
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0]
        gt_colors_lab.append(lab.astype(float))
    ex = np.array(ex_colors_lab); gt = np.array(gt_colors_lab)
    dist = np.linalg.norm(ex[:, None, :] - gt[None, :, :], axis=-1)
    ex_idx, gt_idx = linear_sum_assignment(dist)
    return {int(i): int(j) for i, j in zip(ex_idx, gt_idx)}


def hungarian_xy_error(ex_x, ex_y, gt_x, gt_y, x_span, y_span):
    """Optimal 1:1 assignment of extracted -> GT points; return error stats."""
    if len(ex_x) == 0 or len(gt_x) == 0:
        return {"n_ex": int(len(ex_x)), "n_gt": int(len(gt_x))}
    ex_x = np.asarray(ex_x); ex_y = np.asarray(ex_y)
    gt_x = np.asarray(gt_x); gt_y = np.asarray(gt_y)
    cost = np.sqrt(((ex_x[:, None] - gt_x[None, :]) / max(x_span, 1e-9)) ** 2
                   + ((ex_y[:, None] - gt_y[None, :]) / max(y_span, 1e-9)) ** 2)
    r, c = linear_sum_assignment(cost)
    return {
        "n_ex": int(len(ex_x)), "n_gt": int(len(gt_x)),
        "matched": int(len(r)),
        "x_mae": float(np.abs(ex_x[r] - gt_x[c]).mean()),
        "y_mae": float(np.abs(ex_y[r] - gt_y[c]).mean()),
        "x_mae_norm": float(np.abs(ex_x[r] - gt_x[c]).mean() / max(x_span, 1e-9)),
        "y_mae_norm": float(np.abs(ex_y[r] - gt_y[c]).mean() / max(y_span, 1e-9)),
    }


# ---------------- per-image driver ----------------

def process_one(record: dict, out_dir: Path, meta_dir: Path | None) -> dict:
    image_path = Path(record["image"])
    if "error" in record:
        return {"image": image_path.name, "skipped": record["error"]}
    if not record.get("x_calib") or not record.get("y_calib"):
        return {"image": image_path.name, "skipped": "missing axis calibration"}

    img = cv2.imread(str(image_path))
    if img is None:
        return {"image": image_path.name, "skipped": "cannot read image"}

    plot_bgr, origin = crop_plot(img, record["plot_xyxy"])
    extractor = EXTRACTORS.get(record["chart_class"])
    if extractor is None:
        return {"image": image_path.name, "skipped": f"unknown class {record['chart_class']}"}

    hues = find_series_hues(plot_bgr, max_series=MAX_SERIES)
    if not hues:
        return {"image": image_path.name, "skipped": "no series colors found"}

    plot_shape = plot_bgr.shape[:2]
    # Build a legend-exclusion mask in plot-crop-local coords. Any per-hue mask
    # AND'd with ~legend_mask drops any colored pixels that fall inside the
    # legend rectangle (which often overlays the plot area).
    legend_exclude = None  # bool array of shape plot_shape, True = exclude
    if record.get("legend_xyxy"):
        lx0, ly0, lx1, ly1 = [int(round(v)) for v in record["legend_xyxy"]]
        ox, oy = origin
        plot_h, plot_w = plot_shape
        lx0 = max(0, lx0 - ox); ly0 = max(0, ly0 - oy)
        lx1 = min(plot_w, lx1 - ox); ly1 = min(plot_h, ly1 - oy)
        if lx1 > lx0 and ly1 > ly0:
            legend_exclude = np.zeros(plot_shape, dtype=bool)
            legend_exclude[ly0:ly1, lx0:lx1] = True

    series_results = []
    for s_idx, hue in enumerate(hues):
        mask = hue_mask(plot_bgr, hue)
        if legend_exclude is not None:
            mask = mask & ~legend_exclude
        if mask.sum() < MIN_PIXELS_FOR_SERIES:
            continue
        med_bgr = median_bgr_in_mask(plot_bgr, mask)
        color_hex = bgr_to_hex(med_bgr)
        pixel_pts, markers = extractor(mask, origin, plot_shape)
        if not pixel_pts:
            continue
        data_pts = [pixel_to_data(px, py, record["x_calib"], record["y_calib"])
                    for px, py in pixel_pts]
        series_results.append({
            "series_idx": s_idx,
            "hue": int(hue),
            "color_hex": color_hex,
            "n_points": len(data_pts),
            "markers": markers,
            "pixel_points": pixel_pts,
            "data_points": data_pts,
        })

    # Write CSV
    csv_path = out_dir / f"{image_path.stem}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "color_hex", "marker_guess", "x", "y", "pixel_x", "pixel_y"])
        for s in series_results:
            for (dx, dy), (px, py), mk in zip(s["data_points"], s["pixel_points"], s["markers"]):
                w.writerow([s["series_idx"], s["color_hex"], mk if mk else "",
                            f"{dx:.4f}", f"{dy:.4f}", f"{px:.1f}", f"{py:.1f}"])

    # GT comparison
    metrics_per_series = None
    overall = None
    if meta_dir is not None:
        gt_path = meta_dir / f"{image_path.stem}.json"
        if gt_path.exists():
            meta = json.loads(gt_path.read_text())
            x_span = float(np.ptp(meta["x_range"]))
            y_span = float(np.ptp(meta["y_range"]))
            match = match_series_by_color(series_results, meta["series"])
            metrics_per_series = []
            all_ex_x, all_ex_y, all_gt_x, all_gt_y = [], [], [], []
            for ei, s in enumerate(series_results):
                gj = match.get(ei)
                if gj is None:
                    continue
                gt = meta["series"][gj]
                ex_x = [p[0] for p in s["data_points"]]
                ex_y = [p[1] for p in s["data_points"]]
                err = hungarian_xy_error(ex_x, ex_y, gt["x"], gt["y"], x_span, y_span)
                err["color_hex_extracted"] = s["color_hex"]
                err["color_hex_gt"] = gt["color"]
                metrics_per_series.append(err)
                all_ex_x.extend(ex_x); all_ex_y.extend(ex_y)
                all_gt_x.extend(gt["x"]); all_gt_y.extend(gt["y"])
            if all_ex_x:
                overall = hungarian_xy_error(all_ex_x, all_ex_y, all_gt_x, all_gt_y,
                                             x_span, y_span)

    return {
        "image": image_path.name,
        "chart_class": record["chart_class"],
        "n_series_extracted": len(series_results),
        "n_points_total": sum(s["n_points"] for s in series_results),
        "csv": str(csv_path),
        "metrics_per_series": metrics_per_series,
        "metrics_overall": overall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase4-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--meta-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(Path(args.phase4_json).read_text())
    meta_dir = Path(args.meta_dir) if args.meta_dir else None

    summary = []
    for rec in records:
        s = process_one(rec, out_dir, meta_dir)
        summary.append(s)
        if "skipped" in s:
            print(f"  SKIP {s['image']:35s} {s['skipped']}")
            continue
        m = s.get("metrics_overall") or {}
        bits = (
            f" overall: matched={m.get('matched','?')}/{m.get('n_gt','?')}"
            f" x_mae/sp={m.get('x_mae_norm', float('nan')):.3f}"
            f" y_mae/sp={m.get('y_mae_norm', float('nan')):.3f}"
            if m else ""
        )
        print(f"  OK   {s['image']:35s} {s['chart_class']:13s} "
              f"series_ex={s['n_series_extracted']} pts={s['n_points_total']:3d}{bits}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {len(summary)} records to {out_dir}/summary.json")


if __name__ == "__main__":
    sys.exit(main())
