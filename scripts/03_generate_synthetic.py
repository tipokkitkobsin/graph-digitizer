"""
Phase 2 — generate a diverse synthetic chart dataset.

v2 additions (over v1):
  * NEW class `line_with_scatter` (idx 6) — scatter points overlaid with a fit line
  * Per-image **aspect ratio** sampled from a discrete set (square, 4:3, 17:10, 3:4 ...)
  * **Geometric augmentation** baked into every image: random rotation + perspective warp
    (simulates photographed / photocopied charts whose corners aren't square)
  * Default size grown to 400 train + 100 val

Diversity dimensions sampled per image:
  * chart_type   : scatter | line | bar | line_with_scatter
  * size         : one of {(800,600), (800,800), (1024,600), (600,800), (1024,768), (720,720)}
  * n_series     : 1..3 (scatter/line/line_with_scatter); 1 (bar)
  * mpl style    : default | ggplot | seaborn-v0_8-whitegrid | fivethirtyeight | bmh
  * font family  : DejaVu Sans | serif | monospace
  * colors       : sampled from a 12-color palette (one per series)
  * markers      : scatter sampled from {o s ^ v D x + * P}
  * distractors  : 40% prob, one of {axhline, axvline, fill_between, trend, annotate, none}
  * legend       : 70% prob; random location
  * distortion   : ALWAYS applied — rotation in [-8,+8]°, corner shift in [-7%,+7%] of min(W,H)

YOLO label classes (seven):
  0 scatter_plot  1 line_plot  2 bar_plot  3 x_axis  4 y_axis  5 legend  6 line_with_scatter

Outputs (under <out_root>):
  images/train/*.png   labels/train/*.txt
  images/val/*.png     labels/val/*.txt
  images/test/*.png    labels/test/*.txt   # held-out — used by Phase 4/5
  meta/<stem>.json     # ground-truth: per-series data, color RGB, marker, style, distortion
  data.yaml            # Ultralytics dataset config (includes train + val + test)

Default split: 80/10/10 of 3000 images (2400/300/300). Override with
--n-train / --n-val / --n-test.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import yaml


CLASSES = ["scatter_plot", "line_plot", "bar_plot", "x_axis", "y_axis",
           "legend", "line_with_scatter"]
CHART_TYPES = ["scatter", "line", "bar", "line_with_scatter"]
CHART_TO_CLS = {"scatter": 0, "line": 1, "bar": 2, "line_with_scatter": 6}

# 12 distinct colors with high saturation (good for HSV-based separation later).
PALETTE_HEX = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#17becf", "#bcbd22", "#ff1493", "#00ced1", "#daa520",
]

SCATTER_MARKERS = ["o", "s", "^", "v", "D", "x", "+", "*", "P"]

STYLES = [
    "default", "ggplot", "seaborn-v0_8-whitegrid", "fivethirtyeight", "bmh",
]

FONT_FAMILIES = ["DejaVu Sans", "serif", "monospace"]

LEGEND_LOCS = ["upper right", "upper left", "lower right", "lower left",
               "center right", "best"]

DISTRACTORS = ["axhline", "axvline", "fill_between", "trend", "annotate", "none"]
DISTRACTOR_WEIGHTS = [0.10, 0.10, 0.07, 0.08, 0.05, 0.60]

# Aspect-ratio variety (W, H in pixels). All multiples of typical chart sizes.
IMAGE_SIZES = [
    (800, 600),    # 4:3 standard
    (800, 800),    # 1:1 square
    (1024, 600),   # 17:10 wide
    (600, 800),    # 3:4 portrait
    (1024, 768),   # 4:3 larger
    (720, 720),    # 1:1 smaller
]

DPI = 100

# Distortion bounds (kept gentle so OCR + extraction still work).
DISTORT_MAX_DEG = 8.0          # rotation: ±8°
DISTORT_CORNER_FRAC = 0.07     # perspective: corner shift ±7% of min(W,H)


# ----------------- bbox math -----------------

def display_to_image_bbox(display_bbox, img_h_px):
    """Matplotlib display y is bottom-up; image pixels are top-down."""
    return (display_bbox.x0, img_h_px - display_bbox.y1,
            display_bbox.x1, img_h_px - display_bbox.y0)


def to_yolo(bbox, img_w, img_h):
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, min(x0, img_w))
    x1 = max(0.0, min(x1, img_w))
    y0 = max(0.0, min(y0, img_h))
    y1 = max(0.0, min(y1, img_h))
    return (((x0 + x1) / 2.0) / img_w, ((y0 + y1) / 2.0) / img_h,
            (x1 - x0) / img_w, (y1 - y0) / img_h)


# ----------------- distortion -----------------

def make_distortion_matrix(img_w: int, img_h: int, rng) -> tuple[np.ndarray, dict]:
    """Random rotation + perspective warp as a single 3x3 matrix.

    Returns (M, meta) where meta records the rotation angle and corner shifts
    so we can save them in the per-image meta JSON for debugging.
    """
    # 1) Rotation about image center
    angle = float(rng.uniform(-DISTORT_MAX_DEG, DISTORT_MAX_DEG))
    R2 = cv2.getRotationMatrix2D((img_w / 2.0, img_h / 2.0), angle, 1.0)
    R = np.vstack([R2, [0, 0, 1]]).astype(np.float32)

    # 2) Perspective: shift each of the four image corners independently
    max_shift = DISTORT_CORNER_FRAC * float(min(img_w, img_h))
    src = np.float32([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]])
    corner_shifts = rng.uniform(-max_shift, max_shift, size=(4, 2)).astype(np.float32)
    dst = src + corner_shifts
    P = cv2.getPerspectiveTransform(src, dst)

    M = (P @ R).astype(np.float32)
    return M, {
        "rotation_deg": angle,
        "corner_shift_px": corner_shifts.tolist(),
    }


def warp_image(img_bgr: np.ndarray, M: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    return cv2.warpPerspective(
        img_bgr, M, (img_w, img_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),  # white pad
    )


def warp_bbox(bbox, M: np.ndarray, img_w: int, img_h: int):
    """Transform xyxy bbox -> axis-aligned bbox of the warped corners."""
    x0, y0, x1, y1 = bbox
    corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
    nx0 = max(0.0, float(warped[:, 0].min()))
    ny0 = max(0.0, float(warped[:, 1].min()))
    nx1 = min(float(img_w), float(warped[:, 0].max()))
    ny1 = min(float(img_h), float(warped[:, 1].max()))
    return (nx0, ny0, nx1, ny1)


def fig_to_bgr(fig: plt.Figure) -> np.ndarray:
    """Render a Matplotlib figure to a BGR numpy array (OpenCV order)."""
    fig.canvas.draw()
    # canvas.buffer_rgba is the modern, portable API
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


# ----------------- chart generation -----------------

def sample_series_count(chart_type: str, rng) -> int:
    if chart_type == "bar":
        return 1
    return int(rng.choice([1, 1, 2, 2, 3]))


def sample_data_scatter(rng):
    n = int(rng.integers(15, 40))
    slope = rng.uniform(-2.5, 2.5)
    intercept = rng.uniform(-5, 5)
    noise = rng.uniform(0.5, 2.5)
    x = np.sort(rng.uniform(0, 10, n))
    y = slope * x + intercept + rng.normal(0, noise, n)
    return x, y


def sample_data_line(rng):
    n = int(rng.integers(40, 80))
    x = np.linspace(0, 10, n)
    f = rng.choice(["sin", "cos", "lin", "quad", "exp_decay"])
    if f == "sin":
        y = np.sin(x + rng.uniform(0, 1)) * rng.uniform(0.5, 2) + 0.3 * x
    elif f == "cos":
        y = np.cos(x * rng.uniform(0.5, 1.5)) * rng.uniform(0.5, 2)
    elif f == "lin":
        y = rng.uniform(-1, 1) * x + rng.uniform(-2, 2)
    elif f == "quad":
        y = rng.uniform(0.05, 0.25) * (x - rng.uniform(0, 10)) ** 2 + rng.uniform(-3, 3)
    else:  # exp_decay
        y = rng.uniform(2, 5) * np.exp(-x / rng.uniform(2, 5))
    y = y + rng.normal(0, 0.1, n)
    return x, y


def sample_data_bar(rng):
    n = int(rng.integers(4, 10))
    x = np.arange(n)
    y = rng.uniform(1.0, 10.0, n)
    return x, y


def sample_data_line_with_scatter(rng):
    """Raw scatter points + a smooth underlying trend (returned as
    (x_pts, y_pts, x_fit, y_fit) — the fit is used for plotting only)."""
    n = int(rng.integers(25, 45))
    x = np.sort(rng.uniform(0, 10, n))
    f = rng.choice(["lin", "quad", "sin", "exp_decay"])
    if f == "lin":
        slope = rng.uniform(-2, 2)
        intercept = rng.uniform(-3, 3)
        trend = slope * x + intercept
    elif f == "quad":
        trend = rng.uniform(0.05, 0.25) * (x - rng.uniform(0, 10)) ** 2 + rng.uniform(-3, 3)
    elif f == "sin":
        trend = np.sin(x + rng.uniform(0, 1)) * rng.uniform(0.7, 1.8) + 0.2 * x
    else:  # exp_decay
        trend = rng.uniform(2, 5) * np.exp(-x / rng.uniform(2, 5))
    noise = rng.uniform(0.3, 1.5)
    y = trend + rng.normal(0, noise, n)
    # Fit curve: use polyfit on raw data so the line truly matches what was plotted
    deg = 2 if f in ("quad", "sin") else 1
    if f == "sin":
        deg = 3  # cubic captures the wave roughly
    coeffs = np.polyfit(x, y, deg)
    x_fit = np.linspace(x.min(), x.max(), 80)
    y_fit = np.polyval(coeffs, x_fit)
    return x, y, x_fit, y_fit


def render_one(seed: int) -> tuple[plt.Figure, dict, int, int]:
    rng = np.random.default_rng(seed)

    chart_type = rng.choice(CHART_TYPES)
    img_w, img_h = map(int, IMAGE_SIZES[int(rng.integers(0, len(IMAGE_SIZES)))])
    style = rng.choice(STYLES)
    font_family = rng.choice(FONT_FAMILIES)
    n_series = sample_series_count(chart_type, rng)
    colors = rng.choice(PALETTE_HEX, size=n_series, replace=False).tolist()

    with plt.style.context(style):
        plt.rcParams["font.family"] = font_family
        plt.rcParams["font.size"] = float(rng.uniform(9.5, 12.5))

        fig = plt.figure(figsize=(img_w / DPI, img_h / DPI), dpi=DPI)
        ax = fig.add_subplot(111)

        series_meta = []
        for i in range(n_series):
            label = f"series {i + 1}"
            color = colors[i]

            if chart_type == "scatter":
                marker = str(rng.choice(SCATTER_MARKERS))
                x, y = sample_data_scatter(rng)
                ax.scatter(x, y, marker=marker, s=float(rng.uniform(30, 70)),
                           c=color, edgecolors="none", alpha=0.9, label=label)
                series_meta.append({"label": label, "color": color, "marker": marker,
                                    "x": x.tolist(), "y": y.tolist()})

            elif chart_type == "line":
                lw = float(rng.uniform(1.3, 2.5))
                ls = str(rng.choice(["-", "--", "-.", ":"]))
                x, y = sample_data_line(rng)
                ax.plot(x, y, color=color, linewidth=lw, linestyle=ls, label=label)
                series_meta.append({"label": label, "color": color, "marker": None,
                                    "linestyle": ls, "x": x.tolist(), "y": y.tolist()})

            elif chart_type == "bar":
                x, y = sample_data_bar(rng)
                ax.bar(x, y, color=color, label=label)
                series_meta.append({"label": label, "color": color, "marker": None,
                                    "x": x.tolist(), "y": y.tolist()})

            else:  # line_with_scatter
                marker = str(rng.choice(SCATTER_MARKERS))
                x, y, x_fit, y_fit = sample_data_line_with_scatter(rng)
                # Plot fit line FIRST so scatter sits on top (matches how researchers usually overlay)
                ax.plot(x_fit, y_fit, color=color, linewidth=1.8, alpha=0.85)
                ax.scatter(x, y, marker=marker, s=float(rng.uniform(35, 65)),
                           c=color, edgecolors="none", alpha=0.95, label=label)
                series_meta.append({"label": label, "color": color, "marker": marker,
                                    "x": x.tolist(), "y": y.tolist(),
                                    "fit_x": x_fit.tolist(), "fit_y": y_fit.tolist()})

        ax.set_xlabel(rng.choice(["X axis", "Time", "Position", "Index", "Value"]))
        ax.set_ylabel(rng.choice(["Y axis", "Amplitude", "Count", "Measurement", "Score"]))
        type_pretty = chart_type.replace("_", " ")
        ax.set_title(f"Synthetic {type_pretty} ({n_series} series)" if n_series > 1
                     else f"Synthetic {type_pretty}")

        # Distractors (non-data elements)
        distractor = str(rng.choice(DISTRACTORS, p=DISTRACTOR_WEIGHTS))
        if distractor == "axhline":
            ax.axhline(y=float(rng.uniform(*ax.get_ylim())),
                       color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
        elif distractor == "axvline":
            ax.axvline(x=float(rng.uniform(*ax.get_xlim())),
                       color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
        elif distractor == "fill_between":
            xlim = ax.get_xlim()
            xs = np.linspace(*xlim, 50)
            mid = rng.uniform(*ax.get_ylim())
            band = rng.uniform(0.5, 2.0)
            ax.fill_between(xs, mid - band, mid + band, color="gray", alpha=0.15)
        elif distractor == "trend":
            sx = np.array(series_meta[0]["x"])
            sy = np.array(series_meta[0]["y"])
            if sx.size >= 2 and np.ptp(sx) > 0:
                a, b = np.polyfit(sx, sy, 1)
                xs = np.linspace(sx.min(), sx.max(), 30)
                ax.plot(xs, a * xs + b, color="black", linewidth=1.0,
                        linestyle=":", alpha=0.5)
        elif distractor == "annotate":
            xlim = ax.get_xlim(); ylim = ax.get_ylim()
            ax.text(float(rng.uniform(*xlim)), float(rng.uniform(*ylim)),
                    str(rng.choice(["note", "peak", "ref", "outlier"])),
                    fontsize=9, color="dimgray", alpha=0.7)

        if rng.uniform() < 0.5:
            ax.grid(True, alpha=0.3)

        add_legend = bool(rng.uniform() < 0.7)
        if add_legend:
            ax.legend(loc=str(rng.choice(LEGEND_LOCS)), fontsize=9)

        fig.canvas.draw()

    meta = {
        "chart_type": chart_type,
        "img_w": img_w,
        "img_h": img_h,
        "style": style,
        "font_family": font_family,
        "n_series": n_series,
        "series": series_meta,
        "distractor": distractor,
        "x_range": list(map(float, ax.get_xlim())),
        "y_range": list(map(float, ax.get_ylim())),
        "x_ticks": list(map(float, ax.get_xticks())),
        "y_ticks": list(map(float, ax.get_yticks())),
        "has_legend": add_legend,
    }
    return fig, meta, img_w, img_h


def compute_bboxes(fig: plt.Figure, has_legend: bool, img_h: int) -> dict:
    ax = fig.axes[0]
    renderer = fig.canvas.get_renderer()
    bboxes = {
        "plot": display_to_image_bbox(ax.get_window_extent(renderer), img_h),
        "x_axis": display_to_image_bbox(ax.xaxis.get_tightbbox(renderer), img_h),
        "y_axis": display_to_image_bbox(ax.yaxis.get_tightbbox(renderer), img_h),
    }
    if has_legend:
        legend = ax.get_legend()
        if legend is not None:
            bboxes["legend"] = display_to_image_bbox(
                legend.get_window_extent(renderer), img_h)
    return bboxes


def write_label(label_path: Path, chart_type: str, bboxes: dict,
                img_w: int, img_h: int):
    lines = []
    cls = CHART_TO_CLS[chart_type]
    lines.append(("{} " + "{:.6f} " * 4).format(cls, *to_yolo(bboxes["plot"], img_w, img_h)))
    lines.append(("3 " + "{:.6f} " * 4).format(*to_yolo(bboxes["x_axis"], img_w, img_h)))
    lines.append(("4 " + "{:.6f} " * 4).format(*to_yolo(bboxes["y_axis"], img_w, img_h)))
    if "legend" in bboxes:
        lines.append(("5 " + "{:.6f} " * 4).format(*to_yolo(bboxes["legend"], img_w, img_h)))
    label_path.write_text("\n".join(s.rstrip() for s in lines) + "\n")


def generate_split(split: str, count: int, out_root: Path, base_seed: int):
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    meta_dir = out_root / "meta"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    type_counts = {ct: 0 for ct in CHART_TYPES}
    multi_series_count = 0
    size_counts: dict[tuple[int, int], int] = {}

    for i in range(count):
        seed = base_seed + i
        fig, meta, img_w, img_h = render_one(seed)
        bboxes = compute_bboxes(fig, meta["has_legend"], img_h)

        # Render to BGR, then warp
        img_bgr = fig_to_bgr(fig)
        plt.close(fig)
        M, dist_meta = make_distortion_matrix(img_w, img_h, np.random.default_rng(seed ^ 0xA5A5))
        img_bgr_warped = warp_image(img_bgr, M, img_w, img_h)
        bboxes_warped = {k: warp_bbox(v, M, img_w, img_h) for k, v in bboxes.items()}

        meta["distortion"] = dist_meta
        meta["bboxes_xyxy"] = {k: list(map(float, v)) for k, v in bboxes_warped.items()}

        type_counts[meta["chart_type"]] += 1
        if meta["n_series"] > 1:
            multi_series_count += 1
        size_counts[(img_w, img_h)] = size_counts.get((img_w, img_h), 0) + 1

        stem = f"{split}_{i:04d}_{meta['chart_type']}_n{meta['n_series']}"
        cv2.imwrite(str(img_dir / f"{stem}.png"), img_bgr_warped)
        write_label(lbl_dir / f"{stem}.txt", meta["chart_type"], bboxes_warped,
                    img_w, img_h)
        (meta_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))

    chart_summary = ", ".join(f"{k}={v}" for k, v in type_counts.items())
    size_summary = ", ".join(
        f"{w}x{h}={n}" for (w, h), n in sorted(size_counts.items())
    )
    print(f"  [{split}] {count} images")
    print(f"    chart types  : {chart_summary}")
    print(f"    multi-series : {multi_series_count}")
    print(f"    sizes        : {size_summary}")


def write_data_yaml(out_root: Path, has_test: bool):
    cfg = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASSES)},
    }
    if has_test:
        cfg["test"] = "images/test"
    (out_root / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=2400)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=300,
                    help="Held-out split used by Phase 4/5; 0 disables")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-distortion", action="store_true",
                    help="Skip rotation + perspective warp (cleaner images, "
                         "useful for slide examples or debugging)")
    args = ap.parse_args()

    out_root = Path(args.out)
    # Wipe stale split data so reruns don't mix old + new images
    for sub in ("images/train", "images/val", "images/test",
                "labels/train", "labels/val", "labels/test", "meta"):
        p = out_root / sub
        if p.exists():
            for f in p.iterdir():
                f.unlink()
    out_root.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.no_distortion:
        # Make warp matrix = identity so the image + bboxes pass through
        # unchanged. Cheaper than threading a flag through generate_split.
        global make_distortion_matrix
        def make_distortion_matrix(img_w, img_h, rng):
            return (np.eye(3, dtype=np.float32),
                    {"rotation_deg": 0.0, "corner_shift_px": [[0, 0]] * 4})

    print(f"==> Generating dataset at {out_root}")
    generate_split("train", args.n_train, out_root, base_seed=args.seed)
    generate_split("val", args.n_val, out_root, base_seed=args.seed + 100_000)
    has_test = args.n_test > 0
    if has_test:
        generate_split("test", args.n_test, out_root, base_seed=args.seed + 200_000)
    write_data_yaml(out_root, has_test=has_test)
    parts = [f"{args.n_train} train", f"{args.n_val} val"]
    if has_test:
        parts.append(f"{args.n_test} test")
    print(f"\nDataset ready: {' + '.join(parts)}")
    print(f"  data.yaml at: {out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
