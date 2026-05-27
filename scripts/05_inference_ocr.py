"""
Phase 4 — inference + axis calibration.

Pipeline per image:
  1. YOLOv12s detects plot / x_axis / y_axis / legend boxes.
  2. Pick highest-confidence box for each class.
  3. Crop x_axis and y_axis regions; OCR the tick labels with EasyOCR.
  4. Parse numeric tick values; the min and max OCR'd numbers bound the axis.
  5. Build a linear pixel <-> data mapping for both axes.

If a needed detection is missing (low confidence on the smoke-trained model),
fall back to a heuristic: x_axis = strip below plot bbox, y_axis = strip left.

Output: per-image JSON with detections + calibration; printed summary.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import easyocr
import numpy as np
import torch
from ultralytics import YOLO

CLASSES = ["scatter_plot", "line_plot", "bar_plot", "x_axis", "y_axis",
           "legend", "line_with_scatter"]
PLOT_CLASS_IDS = {0, 1, 2, 6}  # any of these IS the plot
LEGEND_CLASS_ID = 5
NUMBER_RE = re.compile(r"^-?\d+\.?\d*$")


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def best_per_class(boxes_xyxy, confs, cls_ids):
    """Return {class_id: (xyxy_tuple, conf)} keeping max-conf box per class."""
    best: dict[int, tuple[tuple, float]] = {}
    for box, c, k in zip(boxes_xyxy, confs, cls_ids):
        k = int(k)
        if k not in best or c > best[k][1]:
            best[k] = (tuple(map(float, box)), float(c))
    return best


def pick_plot_box(best: dict) -> tuple[int, tuple, float] | None:
    """Pick the highest-conf detection across the three plot classes."""
    candidates = [(k, *best[k]) for k in PLOT_CLASS_IDS if k in best]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[2])  # (class_id, xyxy, conf)


def heuristic_x_axis(plot_xyxy, img_w, img_h):
    """Strip below the plot, full width of the plot, down to image bottom."""
    px0, py0, px1, py1 = plot_xyxy
    return (max(0.0, px0 - 10), py1, min(img_w, px1 + 10), img_h)


def heuristic_y_axis(plot_xyxy, img_w, img_h):
    """Strip to the left of the plot, full height of the plot."""
    px0, py0, px1, py1 = plot_xyxy
    return (0.0, max(0.0, py0 - 10), px0, min(img_h, py1 + 10))


def parse_numbers_from_ocr(results) -> list[tuple[float, tuple]]:
    """EasyOCR readtext result -> [(value, bbox_in_crop), ...] for numeric tokens."""
    out = []
    for entry in results:
        bbox, text, conf = entry
        text_clean = text.strip().replace(",", "").replace(" ", "")
        # Common OCR confusions: lowercase l for 1, uppercase O for 0
        text_clean = text_clean.replace("l", "1").replace("O", "0").replace("o", "0")
        if NUMBER_RE.match(text_clean):
            try:
                out.append((float(text_clean), bbox))
            except ValueError:
                pass
    return out


def crop_xyxy(img, xyxy):
    x0, y0, x1, y1 = [int(round(v)) for v in xyxy]
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(img.shape[1], x1); y1 = min(img.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1].copy()


def ocr_crop(ocr, crop, min_side=160):
    """Resize crop so its smaller side >= min_side, then OCR.
    EasyOCR's CRAFT detector under-detects small text; upscaling lifts recall
    on the thin horizontal x-axis strips (~60 px tall) typical of chart images.
    Returns (raw_ocr_result, scale_used). Tick bboxes are returned in CROP
    coordinates, so we divide by scale to map back."""
    if crop is None or crop.size == 0:
        return [], 1.0
    h, w = crop.shape[:2]
    scale = max(1.0, min_side / max(1, min(h, w)))
    if scale > 1.0:
        crop_resized = cv2.resize(crop, (int(round(w * scale)), int(round(h * scale))),
                                  interpolation=cv2.INTER_CUBIC)
    else:
        crop_resized = crop
    raw = ocr.readtext(crop_resized)
    # Scale OCR bbox coords back to original crop space
    scaled_back = []
    for bbox, text, conf in raw:
        bbox_back = [[p[0] / scale, p[1] / scale] for p in bbox]
        scaled_back.append((bbox_back, text, conf))
    return scaled_back, scale


def calibrate_axis(numbers_with_bbox, axis_crop_origin_xy, axis: str):
    """Given OCR'd tick values + their bboxes in CROP coords, compute pixel->data linear fit.
    axis: 'x' (use bbox center-x in image coords) or 'y' (center-y in image coords).
    Returns dict with slope/intercept or None if <2 tick numbers.
    """
    if len(numbers_with_bbox) < 2:
        return None
    ox, oy = axis_crop_origin_xy
    pixels = []
    values = []
    for val, bbox in numbers_with_bbox:
        # EasyOCR bbox is [[x0,y0],[x1,y0],[x1,y1],[x0,y1]] in crop coords
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx = (min(xs) + max(xs)) / 2 + ox
        cy = (min(ys) + max(ys)) / 2 + oy
        pixels.append(cx if axis == "x" else cy)
        values.append(val)
    pixels = np.array(pixels, dtype=float)
    values = np.array(values, dtype=float)
    # Linear fit: data = a * pixel + b
    if np.allclose(pixels, pixels[0]):
        return None
    a, b = np.polyfit(pixels, values, 1)
    return {
        "slope": float(a),
        "intercept": float(b),
        "n_ticks": int(len(values)),
        "ticks_pixel": pixels.tolist(),
        "ticks_value": values.tolist(),
    }


def process_image(image_path: Path, model: YOLO, ocr: easyocr.Reader, conf_thr: float):
    img = cv2.imread(str(image_path))
    if img is None:
        return {"error": f"cannot read {image_path}"}
    img_h, img_w = img.shape[:2]

    pred = model.predict(source=str(image_path), conf=0.05, verbose=False)[0]
    boxes = pred.boxes.xyxy.cpu().numpy() if pred.boxes is not None else np.zeros((0, 4))
    confs = pred.boxes.conf.cpu().numpy() if pred.boxes is not None else np.zeros((0,))
    clses = pred.boxes.cls.cpu().numpy() if pred.boxes is not None else np.zeros((0,))

    best = best_per_class(boxes, confs, clses)
    plot_pick = pick_plot_box(best)
    if plot_pick is None:
        return {"error": "no plot detected", "boxes": int(len(boxes))}
    plot_cls, plot_xyxy, plot_conf = plot_pick

    used_fallback = {"x_axis": False, "y_axis": False}
    if 3 in best and best[3][1] >= conf_thr:
        x_axis_xyxy, x_axis_conf = best[3]
    else:
        x_axis_xyxy = heuristic_x_axis(plot_xyxy, img_w, img_h)
        x_axis_conf = None
        used_fallback["x_axis"] = True

    if 4 in best and best[4][1] >= conf_thr:
        y_axis_xyxy, y_axis_conf = best[4]
    else:
        y_axis_xyxy = heuristic_y_axis(plot_xyxy, img_w, img_h)
        y_axis_conf = None
        used_fallback["y_axis"] = True

    # Legend bbox (used by Phase 5 to exclude that region from point extraction).
    # None if no legend detected at sufficient confidence.
    if LEGEND_CLASS_ID in best and best[LEGEND_CLASS_ID][1] >= conf_thr:
        legend_xyxy, legend_conf = best[LEGEND_CLASS_ID]
    else:
        legend_xyxy, legend_conf = None, None

    # OCR (with upscaling for small text)
    x_crop = crop_xyxy(img, x_axis_xyxy)
    y_crop = crop_xyxy(img, y_axis_xyxy)
    x_ocr_raw, _ = ocr_crop(ocr, x_crop)
    y_ocr_raw, _ = ocr_crop(ocr, y_crop)
    x_numbers = parse_numbers_from_ocr(x_ocr_raw)
    y_numbers = parse_numbers_from_ocr(y_ocr_raw)

    x_calib = calibrate_axis(x_numbers, (x_axis_xyxy[0], x_axis_xyxy[1]), axis="x")
    y_calib = calibrate_axis(y_numbers, (y_axis_xyxy[0], y_axis_xyxy[1]), axis="y")

    return {
        "image": str(image_path),
        "img_w": img_w, "img_h": img_h,
        "chart_class": CLASSES[plot_cls],
        "plot_xyxy": plot_xyxy, "plot_conf": plot_conf,
        "x_axis_xyxy": x_axis_xyxy, "x_axis_conf": x_axis_conf,
        "y_axis_xyxy": y_axis_xyxy, "y_axis_conf": y_axis_conf,
        "legend_xyxy": legend_xyxy, "legend_conf": legend_conf,
        "used_fallback": used_fallback,
        "x_numbers_ocr": [v for v, _ in x_numbers],
        "y_numbers_ocr": [v for v, _ in y_numbers],
        "x_calib": x_calib,
        "y_calib": y_calib,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Trained best.pt")
    ap.add_argument("--source", required=True,
                    help="Image file OR directory of images")
    ap.add_argument("--out-json", default=None,
                    help="Optional: write a JSON list of all per-image results")
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf-thr", type=float, default=0.25)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"Device: {device}")

    model = YOLO(args.weights)
    # EasyOCR uses CUDA via gpu=True (no MPS support yet); CPU is fine for a smoke test
    use_gpu = device.isdigit() or device == "cuda"
    ocr = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)

    src = Path(args.source)
    if src.is_dir():
        images = sorted([p for p in src.iterdir()
                         if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    else:
        images = [src]
    print(f"Found {len(images)} image(s).")

    results = []
    for p in images:
        print(f"\n--- {p.name} ---")
        r = process_image(p, model, ocr, args.conf_thr)
        results.append(r)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        print(f"  chart={r['chart_class']} conf={r['plot_conf']:.3f}")
        print(f"  plot bbox  = {tuple(round(v, 1) for v in r['plot_xyxy'])}")
        print(f"  x_axis     = {tuple(round(v, 1) for v in r['x_axis_xyxy'])} "
              f"(fallback={r['used_fallback']['x_axis']})")
        print(f"  y_axis     = {tuple(round(v, 1) for v in r['y_axis_xyxy'])} "
              f"(fallback={r['used_fallback']['y_axis']})")
        if r.get("legend_xyxy"):
            print(f"  legend     = {tuple(round(v, 1) for v in r['legend_xyxy'])} "
                  f"(conf={r.get('legend_conf'):.3f})")
        else:
            print("  legend     = (not detected; no exclusion will apply)")
        print(f"  x OCR nums = {r['x_numbers_ocr']}")
        print(f"  y OCR nums = {r['y_numbers_ocr']}")
        if r["x_calib"]:
            print(f"  x calib    = {r['x_calib']['slope']:.4f}*px + {r['x_calib']['intercept']:.4f} "
                  f"(n={r['x_calib']['n_ticks']})")
        else:
            print("  x calib    = INSUFFICIENT TICKS")
        if r["y_calib"]:
            print(f"  y calib    = {r['y_calib']['slope']:.4f}*py + {r['y_calib']['intercept']:.4f} "
                  f"(n={r['y_calib']['n_ticks']})")
        else:
            print("  y calib    = INSUFFICIENT TICKS")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    sys.exit(main())
