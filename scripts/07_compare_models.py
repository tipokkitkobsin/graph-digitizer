"""
Aggregate per-model results into a single comparison CSV.

Inputs:
  --runs    directory containing Ultralytics run folders (each has results.csv)
  --output  directory containing per-model Phase 5 csv_<MODEL>/summary.json
  --prefix  run-name prefix (default 'cmp') to filter runs to compare

Output CSV columns:
  model, run_dir, train_epochs, val_mAP50, val_mAP50_95,
  test_n_images, test_n_extracted_total, test_n_skipped,
  test_x_mae_norm_median, test_y_mae_norm_median,
  test_x_mae_norm_mean, test_y_mae_norm_mean,
  best_pt_size_mb
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def find_col(df: pd.DataFrame, *needles: str) -> str | None:
    for c in df.columns:
        for n in needles:
            if n in c:
                return c
    return None


def read_run_metrics(run_dir: Path) -> dict:
    """Pull final-epoch val mAP from Ultralytics' results.csv."""
    out = {"train_epochs": None, "val_mAP50": None, "val_mAP50_95": None}
    csv = run_dir / "results.csv"
    if not csv.exists():
        return out
    try:
        df = pd.read_csv(csv)
    except Exception:
        return out
    if df.empty:
        return out
    out["train_epochs"] = int(len(df))
    mp50_col = find_col(df, "mAP50(B)", "mAP_0.5", "metrics/mAP50")
    mp_col = find_col(df, "mAP50-95(B)", "mAP_0.5:0.95", "metrics/mAP50-95")
    if mp50_col:
        out["val_mAP50"] = float(df[mp50_col].iloc[-1])
    if mp_col:
        out["val_mAP50_95"] = float(df[mp_col].iloc[-1])
    return out


def read_extraction_metrics(summary_json: Path) -> dict:
    """Aggregate per-image Phase 5 errors from csv_<MODEL>/summary.json."""
    out = {
        "test_n_images": 0,
        "test_n_extracted_total": 0,
        "test_n_skipped": 0,
        "test_x_mae_norm_median": None,
        "test_y_mae_norm_median": None,
        "test_x_mae_norm_mean": None,
        "test_y_mae_norm_mean": None,
    }
    if not summary_json.exists():
        return out
    try:
        rows = json.loads(summary_json.read_text())
    except Exception:
        return out
    x_errs: list[float] = []
    y_errs: list[float] = []
    for r in rows:
        out["test_n_images"] += 1
        if "skipped" in r:
            out["test_n_skipped"] += 1
            continue
        out["test_n_extracted_total"] += int(r.get("n_points_total", 0))
        m = r.get("metrics_overall") or {}
        x = m.get("x_mae_norm")
        y = m.get("y_mae_norm")
        if isinstance(x, (int, float)) and not math.isnan(x):
            x_errs.append(float(x))
        if isinstance(y, (int, float)) and not math.isnan(y):
            y_errs.append(float(y))
    if x_errs:
        s = pd.Series(x_errs)
        out["test_x_mae_norm_median"] = float(s.median())
        out["test_x_mae_norm_mean"] = float(s.mean())
    if y_errs:
        s = pd.Series(y_errs)
        out["test_y_mae_norm_median"] = float(s.median())
        out["test_y_mae_norm_mean"] = float(s.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prefix", default="cmp",
                    help="Run-name prefix to match (e.g. 'cmp' matches cmp_yolo12n etc.)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs_root = Path(args.runs)
    out_root = Path(args.output)

    rows = []
    run_dirs = sorted(p for p in runs_root.iterdir()
                      if p.is_dir() and p.name.startswith(f"{args.prefix}_"))
    for run_dir in run_dirs:
        model = run_dir.name[len(args.prefix) + 1:]  # strip "cmp_"
        best_pt = run_dir / "weights" / "best.pt"
        size_mb = round(best_pt.stat().st_size / 1024 / 1024, 1) if best_pt.exists() else None

        m = read_run_metrics(run_dir)
        e = read_extraction_metrics(out_root / f"csv_{model}" / "summary.json")
        rows.append({
            "model": model,
            "run_dir": str(run_dir.relative_to(runs_root.parent)),
            "train_epochs": m["train_epochs"],
            "val_mAP50": m["val_mAP50"],
            "val_mAP50_95": m["val_mAP50_95"],
            **e,
            "best_pt_size_mb": size_mb,
        })

    if not rows:
        print(f"No runs found under {runs_root} with prefix '{args.prefix}'")
        return

    df = pd.DataFrame(rows)
    # Round float columns for readability
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # Pretty-print to stdout
    print(f"\nWrote {args.out}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
