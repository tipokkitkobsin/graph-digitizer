"""
Phase 3 — fine-tune a YOLO detector on the chart-component dataset.

Works for any Ultralytics-compatible variant: yolo12{n,s,m,l,x}, yolo26{...},
yolo11, yolov8, etc. Pass the variant via --weights (auto-downloaded by name)
or as an explicit .pt path.

Runs locally on MPS or on Vast.ai with a CUDA GPU (auto-detected). Use --budget
to pick a preset; presets set epochs + batch, plus an Ultralytics early-stop
patience (default 5: stop if val mAP doesn't improve for 5 epochs).

    smoke  :   5 epochs, batch 4, no early stop (~30 s on RTX 4090; sanity check)
    quick  :  10 epochs, batch 8, patience=5   (~2-4 min on RTX 4090)
    medium :  20 epochs, batch 8, patience=5   (~3-8 min on RTX 4090; DEFAULT)
    full   :  50 epochs, batch 8, patience=5   (~5-15 min on RTX 4090)

Usage:
    python scripts/04_train_yolov12.py \
        --data "$WORK/dataset/data.yaml" \
        --weights "$WORK/weights/yolo12s.pt" \
        --project "$WORK/runs" --name diverse-v2
    # (--budget defaults to medium)
"""

from __future__ import annotations

import argparse
import sys

import torch
from ultralytics import YOLO


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to data.yaml")
    ap.add_argument("--weights", default="yolo12s.pt",
                    help="Starting weights — either a .pt path or an ultralytics model name")
    ap.add_argument("--project", default="runs/train", help="Output project dir")
    ap.add_argument("--name", default="chart_yolov12s", help="Run name")
    ap.add_argument("--device", default=None, help="cuda idx / 'mps' / 'cpu' (auto if unset)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override preset epochs")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=None, help="Override preset batch")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=None,
                    help="Early-stop after N epochs without val-mAP improvement (0 disables)")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--budget",
                    choices=["smoke", "medium", "quick", "full"],
                    default="medium",
                    help="Preset: smoke (5 ep) | quick (10 ep) | medium (20 ep, DEFAULT) | full (50 ep)")
    args = ap.parse_args()

    device = pick_device(args.device)
    workers = args.workers
    amp = args.amp

    # MPS + AMP is still flaky in torch 2.10
    if device == "mps":
        amp = False

    # Preset defaults — CLI args win if set.
    presets = {
        "smoke":  dict(epochs=5,  batch=4, patience=0),
        "quick":  dict(epochs=10, batch=8, patience=5),
        "medium": dict(epochs=20, batch=8, patience=5),
        "full":   dict(epochs=50, batch=8, patience=5),
    }
    preset = presets[args.budget]
    epochs   = args.epochs   if args.epochs   is not None else preset["epochs"]
    batch    = args.batch    if args.batch    is not None else preset["batch"]
    patience = args.patience if args.patience is not None else preset["patience"]
    if args.budget == "smoke":
        workers, amp = 0, False
    elif device == "mps":
        workers = 0

    print(f"[budget={args.budget}] epochs={epochs} batch={batch} "
          f"patience={patience} workers={workers} amp={amp}")
    print(f"Device: {device} | weights={args.weights} | imgsz={args.imgsz}")

    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        epochs=epochs,
        imgsz=args.imgsz,
        batch=batch,
        workers=workers,
        device=device,
        amp=amp,
        patience=patience,           # Ultralytics built-in early stop
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    save_dir = getattr(results, "save_dir", None) or model.trainer.save_dir
    best_pt = f"{save_dir}/weights/best.pt"
    last_pt = f"{save_dir}/weights/last.pt"
    print(f"\nTraining complete. best={best_pt}  last={last_pt}")


if __name__ == "__main__":
    sys.exit(main())
