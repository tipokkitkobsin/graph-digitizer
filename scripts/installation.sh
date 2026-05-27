#!/usr/bin/env bash
# One-shot environment bootstrap for the Graph Digitizer pipeline.
#
# Designed for a fresh Vast.ai instance whose base image is:
#   pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
# (any Debian/Ubuntu-derived image with PyTorch + CUDA pre-installed works).
#
# Idempotent — re-running is safe; apt + pip will skip what's already installed.
#
# Usage (after SSH'ing into the instance):
#   cd /workspace/graph-digitizer        # or wherever you uploaded the project
#   bash scripts/installation.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "==> Project root: $REPO_ROOT"

# ---------- [1/4] system deps for OpenCV + EasyOCR + dev convenience ----------
echo ""
echo "==> [1/4] apt deps"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    ffmpeg git curl unzip vim tmux htop ca-certificates
rm -rf /var/lib/apt/lists/*

# ---------- [2/4] pip deps ----------
echo ""
echo "==> [2/4] pip deps (from requirements.txt)"
python -m pip install --no-cache-dir --upgrade pip wheel setuptools
python -m pip install --no-cache-dir -r "$REPO_ROOT/requirements.txt"

# ---------- [3/4] sanity check ----------
echo ""
echo "==> [3/4] sanity check"
python - <<'PY'
import sys
mods = ["torch", "ultralytics", "cv2", "easyocr", "pandas", "skimage",
        "scipy", "yaml", "matplotlib", "numpy"]
bad = []
for m in mods:
    try:
        __import__(m)
        print(f"  ok   {m}")
    except Exception as e:
        bad.append((m, e))
        print(f"  FAIL {m}: {e}")
if bad:
    sys.exit(f"\n{len(bad)} import(s) failed.")
PY

# ---------- [4/4] GPU report ----------
echo ""
echo "==> [4/4] GPU report"
python - <<'PY'
import sys
import torch
print(f"torch          : {torch.__version__}")
print(f"torch.cuda     : {torch.version.cuda}")
print(f"cuda available : {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("GPU not visible. Vast.ai instance has no CUDA device; the pipeline will not be usable.")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    cap = torch.cuda.get_device_capability(i)
    print(f"  [{i}] {p.name} | sm_{cap[0]}{cap[1]} | "
          f"{p.multi_processor_count} SMs | {p.total_memory / (1024**3):.1f} GiB VRAM")
PY

echo ""
echo "==> Installation complete."
echo "    Next: bash scripts/full_run.sh"
