"""
Graph Digitizer pipeline — Phase 4 (detect + axis OCR + calibration) + Phase 5
(per-color multi-series extraction). Self-contained module imported by server.py.

Mirrors webapp/ml-backend/app/pipeline.py from the local Docker stack so the
output schema is identical — any frontend code that knew how to render the
docker app's predict response will work here too.
"""

from __future__ import annotations

import base64
import re
from typing import Any, Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import theilslopes
from skimage.morphology import skeletonize


CLASSES = ["scatter_plot", "line_plot", "bar_plot", "x_axis", "y_axis",
           "legend", "line_with_scatter"]
PLOT_CLASS_IDS = {0, 1, 2, 6}
LEGEND_CLASS_ID = 5
NUMBER_RE = re.compile(r"^-?\d+\.?\d*$")

HUE_BINS = 180
MIN_PIXELS_FOR_SERIES = 25
MAX_SERIES = 5
HUE_PEAK_MIN_DIST = 8
# Lowered from 0.07 to 0.03: under heavy distortion the per-color pixel count
# of a thin line can be one-third the dominant series' pixel count, and the
# original 7% floor was rejecting the weaker series.
MIN_PEAK_FRACTION = 0.03
DISTRACTOR_ASPECT_FRAC = 0.70
DISTRACTOR_MIN_AR = 8.0


# ============ Phase 4: detect + OCR + calibrate ============

def best_per_class(boxes, confs, cls_ids):
    best: dict[int, tuple[tuple, float]] = {}
    for box, c, k in zip(boxes, confs, cls_ids):
        k = int(k)
        if k not in best or c > best[k][1]:
            best[k] = (tuple(map(float, box)), float(c))
    return best


def pick_plot_box(best):
    cands = [(k, *best[k]) for k in PLOT_CLASS_IDS if k in best]
    if not cands:
        return None
    return max(cands, key=lambda t: t[2])  # (class_id, xyxy, conf)


def heuristic_x_axis(plot_xyxy, img_w, img_h):
    px0, py0, px1, py1 = plot_xyxy
    return (max(0.0, px0 - 10), py1, min(img_w, px1 + 10), img_h)


def heuristic_y_axis(plot_xyxy, img_w, img_h):
    px0, py0, px1, py1 = plot_xyxy
    return (0.0, max(0.0, py0 - 10), px0, min(img_h, py1 + 10))


def crop_xyxy(img, xyxy):
    x0, y0, x1, y1 = [int(round(v)) for v in xyxy]
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(img.shape[1], x1); y1 = min(img.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1].copy()


def ocr_crop(ocr, crop, min_side=240):
    """Upscale crop to min_side on its smaller dimension before OCR.
    Larger crops give EasyOCR's CRAFT detector more pixels to work with on
    small tick labels. Default 240 (was 160 in the HF-optimized build)."""
    if crop is None or crop.size == 0:
        return []
    h, w = crop.shape[:2]
    scale = max(1.0, min_side / max(1, min(h, w)))
    if scale > 1.0:
        crop = cv2.resize(crop, (int(round(w * scale)), int(round(h * scale))),
                          interpolation=cv2.INTER_CUBIC)
    raw = ocr.readtext(crop)
    return [([[p[0] / scale, p[1] / scale] for p in bb], text, conf)
            for bb, text, conf in raw]


def parse_numbers(results):
    out = []
    for bbox, text, _conf in results:
        t = text.strip().replace(",", "").replace(" ", "")
        t = t.replace("l", "1").replace("O", "0").replace("o", "0")
        if NUMBER_RE.match(t):
            try:
                out.append((float(t), bbox))
            except ValueError:
                pass
    return out


def calibrate_axis(numbers, origin_xy, axis):
    """Linear fit pixel -> data with robust regression when possible.

    With 3+ OCR'd tick numbers we use **Theil-Sen** estimation (median of
    pairwise slopes), which tolerates up to ~29% outliers in y. This survives
    common OCR misreads like '55' for '5' that wreck a least-squares polyfit.
    Falls back to standard polyfit for exactly 2 ticks.
    """
    if len(numbers) < 2:
        return None
    ox, oy = origin_xy
    pixels, values = [], []
    for val, bbox in numbers:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx = (min(xs) + max(xs)) / 2 + ox
        cy = (min(ys) + max(ys)) / 2 + oy
        pixels.append(cx if axis == "x" else cy)
        values.append(val)
    pixels = np.array(pixels, dtype=float)
    values = np.array(values, dtype=float)
    if np.allclose(pixels, pixels[0]):
        return None
    if len(values) >= 3:
        slope, intercept, _lo, _hi = theilslopes(values, pixels)
        method = "theilsen"
    else:
        slope, intercept = np.polyfit(pixels, values, 1)
        method = "polyfit"
    return {
        "slope": float(slope), "intercept": float(intercept),
        "n_ticks": int(len(values)),
        "method": method,
        "ticks_pixel": pixels.tolist(),
        "ticks_value": values.tolist(),
    }


def detect_and_calibrate(image_bgr, model, ocr, conf_thr: float = 0.25) -> dict:
    """Run YOLO + axis OCR. Uses ultralytics' default imgsz (640) because the
    model was trained at imgsz=640 — running inference at a different scale
    misaligns the receptive field and tanks detection quality."""
    if image_bgr is None:
        return {"error": "no image"}
    img_h, img_w = image_bgr.shape[:2]

    pred = model.predict(source=image_bgr, conf=0.05, verbose=False)[0]
    boxes = pred.boxes.xyxy.cpu().numpy() if pred.boxes is not None else np.zeros((0, 4))
    confs = pred.boxes.conf.cpu().numpy() if pred.boxes is not None else np.zeros((0,))
    clses = pred.boxes.cls.cpu().numpy() if pred.boxes is not None else np.zeros((0,))

    best = best_per_class(boxes, confs, clses)
    plot_pick = pick_plot_box(best)
    if plot_pick is None:
        return {"error": "no plot detected", "n_boxes": int(len(boxes))}
    plot_cls, plot_xyxy, plot_conf = plot_pick

    used_fallback = {"x_axis": False, "y_axis": False}
    if 3 in best and best[3][1] >= conf_thr:
        x_axis_xyxy = best[3][0]
    else:
        x_axis_xyxy = heuristic_x_axis(plot_xyxy, img_w, img_h)
        used_fallback["x_axis"] = True
    if 4 in best and best[4][1] >= conf_thr:
        y_axis_xyxy = best[4][0]
    else:
        y_axis_xyxy = heuristic_y_axis(plot_xyxy, img_w, img_h)
        used_fallback["y_axis"] = True

    if LEGEND_CLASS_ID in best and best[LEGEND_CLASS_ID][1] >= conf_thr:
        legend_xyxy = list(best[LEGEND_CLASS_ID][0])
        legend_conf = float(best[LEGEND_CLASS_ID][1])
    else:
        legend_xyxy, legend_conf = None, None

    x_crop = crop_xyxy(image_bgr, x_axis_xyxy)
    y_crop = crop_xyxy(image_bgr, y_axis_xyxy)
    x_ocr = ocr_crop(ocr, x_crop)
    y_ocr = ocr_crop(ocr, y_crop)
    x_nums = parse_numbers(x_ocr)
    y_nums = parse_numbers(y_ocr)

    return {
        "img_w": img_w, "img_h": img_h,
        "chart_class": CLASSES[plot_cls],
        "plot_xyxy": list(plot_xyxy), "plot_conf": plot_conf,
        "x_axis_xyxy": list(x_axis_xyxy),
        "y_axis_xyxy": list(y_axis_xyxy),
        "legend_xyxy": legend_xyxy, "legend_conf": legend_conf,
        "used_fallback": used_fallback,
        "x_numbers_ocr": [v for v, _ in x_nums],
        "y_numbers_ocr": [v for v, _ in y_nums],
        "x_calib": calibrate_axis(x_nums, (x_axis_xyxy[0], x_axis_xyxy[1]), "x"),
        "y_calib": calibrate_axis(y_nums, (y_axis_xyxy[0], y_axis_xyxy[1]), "y"),
    }


# ============ Phase 5: per-color extraction ============

def pixel_to_data(px, py, x_calib, y_calib):
    return (float(x_calib["slope"] * px + x_calib["intercept"]),
            float(y_calib["slope"] * py + y_calib["intercept"]))


def bgr_to_hex(bgr):
    b, g, r = [int(v) for v in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def crop_plot(img, xyxy):
    x0, y0, x1, y1 = [int(round(v)) for v in xyxy]
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(img.shape[1], x1); y1 = min(img.shape[0], y1)
    return img[y0:y1, x0:x1].copy(), (x0, y0)


def find_series_hues(plot_bgr, max_series=MAX_SERIES):
    hsv = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2HSV)
    sat_mask = hsv[..., 1] > 80
    val_mask = hsv[..., 2] < 254
    valid = sat_mask & val_mask
    hues = hsv[..., 0][valid]
    if hues.size < MIN_PIXELS_FOR_SERIES:
        return []
    hist, _ = np.histogram(hues, bins=HUE_BINS, range=(0, HUE_BINS))
    smooth = gaussian_filter1d(hist.astype(float), sigma=2.0, mode="wrap")
    # find_peaks can't detect peaks at indices 0 or N-1 (no left/right neighbor).
    # Hue 0 (red) and hue 179 (also red) are the most common boundary colors and
    # were silently dropped. Tile 3x and detect peaks in the middle copy so every
    # original index has full bilateral context, then unwrap back to [0, N).
    n = len(smooth)
    tiled = np.concatenate([smooth, smooth, smooth])
    peaks_t, _ = find_peaks(tiled, height=smooth.max() * MIN_PEAK_FRACTION,
                            distance=HUE_PEAK_MIN_DIST)
    peaks = np.unique(peaks_t[(peaks_t >= n) & (peaks_t < 2 * n)] % n)
    if peaks.size == 0:
        return []
    order = np.argsort(smooth[peaks])[::-1]
    peaks = peaks[order][:max_series]
    return [int(p) for p in peaks]


def hue_mask(plot_bgr, hue_center, half_window=7):
    hsv = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(int)
    sat = hsv[..., 1] > 80
    val = hsv[..., 2] < 254
    d = np.minimum((h - hue_center) % HUE_BINS, (hue_center - h) % HUE_BINS)
    return (d <= half_window) & sat & val


def median_bgr_in_mask(plot_bgr, mask):
    if not mask.any():
        return (0, 0, 0)
    return tuple(int(v) for v in np.median(plot_bgr[mask], axis=0))


def is_distractor(stat, plot_shape):
    w = int(stat[cv2.CC_STAT_WIDTH]); h = int(stat[cv2.CC_STAT_HEIGHT])
    plot_h, plot_w = plot_shape
    if w == 0 or h == 0:
        return False
    ar = max(w / max(1, h), h / max(1, w))
    return ar >= DISTRACTOR_MIN_AR and (w >= DISTRACTOR_ASPECT_FRAC * plot_w
                                        or h >= DISTRACTOR_ASPECT_FRAC * plot_h)


def extract_scatter(mask, origin_xy, plot_shape):
    ox, oy = origin_xy
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    pts, markers = [], []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (8 <= area <= 2500):
            continue
        if is_distractor(stats[i], plot_shape):
            continue
        cnts, _ = cv2.findContours((labels == i).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        marker = "round"
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            peri = cv2.arcLength(c, True)
            circ = 4 * np.pi * area / max(peri ** 2, 1.0)
            marker = "round" if circ > 0.65 else "angular"
        cx, cy = centroids[i]
        pts.append((cx + ox, cy + oy))
        markers.append(marker)
    return pts, markers


def extract_line(mask, origin_xy, plot_shape):
    ox, oy = origin_xy
    if mask.sum() < 30:
        return [], []
    # Anti-aliasing + perspective distortion can shatter a real chart line into
    # 100+ tiny components. Bridge the gaps with morphological closing so the
    # line becomes one continuous component for skeletonization.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bridged = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)
    keep = np.zeros_like(bridged, dtype=bool)
    for i in range(1, n):
        # Do NOT filter by is_distractor() — a real chart line is exactly what
        # is_distractor flags (long, thin, spans the plot width). Reference
        # overlays (axhline, trend lines) are drawn in low-saturation gray and
        # are already excluded upstream by the hue_mask saturation gate.
        if stats[i, cv2.CC_STAT_AREA] < 5:
            continue
        keep |= (labels == i)
    if keep.sum() < 30:
        return [], []
    skel = skeletonize(keep)
    h, w = skel.shape
    pts = []
    for x in range(w):
        ys = np.where(skel[:, x])[0]
        if ys.size:
            pts.append((x + ox, float(np.median(ys)) + oy))
    if len(pts) > 80:
        step = max(1, len(pts) // 80)
        pts = pts[::step]
    return pts, [None] * len(pts)


def extract_bar(mask, origin_xy, plot_shape):
    ox, oy = origin_xy
    m = mask.astype(np.uint8)
    if m.sum() < 30:
        return [], []
    # Bridge tiny anti-aliasing/distortion gaps so each bar's column-wise mass
    # is more solid. Same trick as extract_line.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    col = m.sum(axis=0)
    if col.max() < 3:
        return [], []
    # Adaptive column threshold: relative to the densest column. The original
    # absolute threshold (`plot_h * 0.05`) rejected legitimate bars when
    # perspective distortion made each column thin in the y direction.
    threshold = max(3, col.max() * 0.4)
    in_bar = col > threshold
    pts = []
    i, n = 0, len(in_bar)
    while i < n:
        if in_bar[i]:
            j = i
            while j < n and in_bar[j]:
                j += 1
            row_any = m[:, i:j].any(axis=1)
            top = np.where(row_any)[0]
            if top.size:
                pts.append(((i + j) / 2.0 + ox, float(top.min()) + oy))
            i = j
        else:
            i += 1
    return pts, [None] * len(pts)


def extract_scatter_with_line_filter(mask, origin_xy, plot_shape):
    """Stricter scatter extractor — drops elongated components so the fit line
    overlaid on `line_with_scatter` charts isn't emitted as points."""
    ox, oy = origin_xy
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    pts, markers = [], []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (8 <= area <= 1200):
            continue
        if is_distractor(stats[i], plot_shape):
            continue
        cnts, _ = cv2.findContours((labels == i).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / max(peri ** 2, 1.0)
        if circ < 0.5:
            continue
        cx, cy = centroids[i]
        pts.append((cx + ox, cy + oy))
        markers.append("round" if circ > 0.65 else "angular")
    return pts, markers


EXTRACTORS = {
    "scatter_plot": extract_scatter,
    "line_plot": extract_line,
    "bar_plot": extract_bar,
    "line_with_scatter": extract_scatter_with_line_filter,
}


def extract_series(image_bgr, phase4: dict) -> list[dict]:
    if "error" in phase4 or not phase4.get("x_calib") or not phase4.get("y_calib"):
        return []
    plot_bgr, origin = crop_plot(image_bgr, phase4["plot_xyxy"])
    extractor = EXTRACTORS.get(phase4["chart_class"])
    if extractor is None:
        return []
    hues = find_series_hues(plot_bgr, max_series=MAX_SERIES)
    if not hues:
        return []
    plot_shape = plot_bgr.shape[:2]

    legend_exclude = None
    if phase4.get("legend_xyxy"):
        lx0, ly0, lx1, ly1 = [int(round(v)) for v in phase4["legend_xyxy"]]
        ox, oy = origin
        plot_h, plot_w = plot_shape
        lx0 = max(0, lx0 - ox); ly0 = max(0, ly0 - oy)
        lx1 = min(plot_w, lx1 - ox); ly1 = min(plot_h, ly1 - oy)
        if lx1 > lx0 and ly1 > ly0:
            legend_exclude = np.zeros(plot_shape, dtype=bool)
            legend_exclude[ly0:ly1, lx0:lx1] = True

    out = []
    for s_idx, hue in enumerate(hues):
        mask = hue_mask(plot_bgr, hue)
        if legend_exclude is not None:
            mask = mask & ~legend_exclude
        if mask.sum() < MIN_PIXELS_FOR_SERIES:
            continue
        med = median_bgr_in_mask(plot_bgr, mask)
        color_hex = bgr_to_hex(med)
        pixel_pts, markers = extractor(mask, origin, plot_shape)
        if not pixel_pts:
            continue
        data_pts = [pixel_to_data(px, py, phase4["x_calib"], phase4["y_calib"])
                    for px, py in pixel_pts]
        out.append({
            "series_idx": s_idx,
            "color_hex": color_hex,
            "n_points": len(data_pts),
            "points": [
                {"x": dx, "y": dy, "pixel_x": px, "pixel_y": py, "marker": m}
                for (dx, dy), (px, py), m in zip(data_pts, pixel_pts, markers)
            ],
        })
    return out


# ============ visualization + CSV ============

def annotate_image(image_bgr, phase4: dict, series: list[dict]) -> bytes:
    img = image_bgr.copy()
    if "error" not in phase4 and phase4.get("plot_xyxy"):
        x0, y0, x1, y1 = [int(round(v)) for v in phase4["plot_xyxy"]]
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 0), 2)
        cv2.putText(img, f"{phase4.get('chart_class','?')} {phase4.get('plot_conf',0):.2f}",
                    (x0, max(15, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
    if "error" not in phase4 and phase4.get("legend_xyxy"):
        lx0, ly0, lx1, ly1 = [int(round(v)) for v in phase4["legend_xyxy"]]
        cv2.rectangle(img, (lx0, ly0), (lx1, ly1), (0, 215, 255), 1)
        cv2.putText(img, "legend (excluded)",
                    (lx0, max(15, ly0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (0, 215, 255), 1)
    for s in series:
        h = s["color_hex"].lstrip("#")
        rgb = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        bgr = (rgb[2], rgb[1], rgb[0])
        for p in s["points"]:
            cv2.circle(img, (int(p["pixel_x"]), int(p["pixel_y"])), 5, bgr, 2)
    ok, buf = cv2.imencode(".png", img)
    return bytes(buf) if ok else b""


def series_to_csv(series: list[dict]) -> str:
    lines = ["series,color_hex,marker,x,y,pixel_x,pixel_y"]
    for s in series:
        for p in s["points"]:
            lines.append(
                f"{s['series_idx']},{s['color_hex']},{p.get('marker') or ''},"
                f"{p['x']:.4f},{p['y']:.4f},{p['pixel_x']:.1f},{p['pixel_y']:.1f}"
            )
    return "\n".join(lines) + "\n"
