#!/usr/bin/env bash
# End-to-end multi-model runner: trains 6 detectors (yolo12 n/s/m, yolo26 n/s/m)
# on the same chart-component dataset, runs Phase 4 + Phase 5 on the held-out
# TEST set per model, aggregates results into a comparison table, and bundles
# everything into a single tarball for download.
#
# Idempotent — each model is skipped if its best.pt already exists.
#
# Usage:
#   bash scripts/full_run.sh                              # all defaults
#   MODELS="yolo12s yolo26s" bash scripts/full_run.sh     # subset
#   BUDGET=quick bash scripts/full_run.sh                 # 30 ep instead of 50
#   N_TRAIN=200 N_VAL=25 N_TEST=25 bash scripts/full_run.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

WORK="${WORK:-/workspace/graph-digitizer}"
BUDGET="${BUDGET:-medium}"
N_TRAIN="${N_TRAIN:-2400}"
N_VAL="${N_VAL:-300}"
N_TEST="${N_TEST:-300}"
SEED="${SEED:-42}"
MODELS="${MODELS:-yolo12n yolo12s yolo12m yolo26n yolo26s yolo26m}"
RUN_PREFIX="${RUN_PREFIX:-cmp}"

echo "==========================================================="
echo "  Graph Digitizer multi-model run"
echo "==========================================================="
echo "  REPO_ROOT : $REPO_ROOT"
echo "  WORK      : $WORK"
echo "  BUDGET    : $BUDGET"
echo "  DATASET   : $N_TRAIN train + $N_VAL val + $N_TEST test  (seed $SEED)"
echo "  MODELS    : $MODELS"
echo ""

mkdir -p "$WORK"/{dataset,weights,runs,output}

# ============================================================
# [1/5] Dataset
# ============================================================
echo "==> [1/5] Dataset"
REGEN=0
if [[ -f "$WORK/dataset/data.yaml" && -d "$WORK/dataset/images/test" ]]; then
    n_train_actual=$(ls "$WORK/dataset/images/train" 2>/dev/null | wc -l | tr -d ' ')
    n_val_actual=$(ls   "$WORK/dataset/images/val"   2>/dev/null | wc -l | tr -d ' ')
    n_test_actual=$(ls  "$WORK/dataset/images/test"  2>/dev/null | wc -l | tr -d ' ')
    if [[ "$n_train_actual" == "$N_TRAIN" && "$n_val_actual" == "$N_VAL" \
          && "$n_test_actual" == "$N_TEST" ]]; then
        echo "    Reusing existing dataset: $n_train_actual / $n_val_actual / $n_test_actual"
    else
        echo "    Existing dataset is $n_train_actual / $n_val_actual / $n_test_actual"
        echo "    Requested:           $N_TRAIN / $N_VAL / $N_TEST  — regenerating."
        REGEN=1
    fi
else
    if [[ -f "$WORK/dataset/data.yaml" ]]; then
        echo "    Existing dataset has no test/ split — regenerating with 80/10/10."
    fi
    REGEN=1
fi
if [[ "$REGEN" == "1" ]]; then
    python "$REPO_ROOT/scripts/03_generate_synthetic.py" \
        --out "$WORK/dataset" \
        --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED"
fi

# ============================================================
# [2/5] Pre-download all pretrained weights once
# ============================================================
echo ""
echo "==> [2/5] Pretrained weights"
(
    cd "$WORK/weights"
    for MODEL in $MODELS; do
        if [[ -f "${MODEL}.pt" ]]; then
            echo "    ${MODEL}.pt already present"
        else
            echo "    fetching ${MODEL}.pt"
            python -c "from ultralytics import YOLO; YOLO('${MODEL}.pt')"
        fi
    done
)

# ============================================================
# [3/5] Train each model
# ============================================================
echo ""
echo "==> [3/5] Training (budget=$BUDGET)"
for MODEL in $MODELS; do
    RUN_NAME="${RUN_PREFIX}_${MODEL}"
    BEST="$WORK/runs/$RUN_NAME/weights/best.pt"
    if [[ -f "$BEST" ]]; then
        echo "    [$MODEL] skipping — best.pt already exists at $BEST"
        continue
    fi
    echo ""
    echo "    --- [$MODEL] training ---"
    python "$REPO_ROOT/scripts/04_train_yolov12.py" \
        --data "$WORK/dataset/data.yaml" \
        --weights "$WORK/weights/${MODEL}.pt" \
        --project "$WORK/runs" --name "$RUN_NAME" \
        --budget "$BUDGET"
done

# ============================================================
# [4/5] Phase 4 inference + Phase 5 extraction (on TEST split, per model)
# ============================================================
echo ""
echo "==> [4/5] Per-model Phase 4 + Phase 5 (on test split)"
TEST_DIR="$WORK/dataset/images/test"
if [[ ! -d "$TEST_DIR" ]]; then
    echo "    test/ split missing; falling back to val/"
    TEST_DIR="$WORK/dataset/images/val"
fi

for MODEL in $MODELS; do
    RUN_NAME="${RUN_PREFIX}_${MODEL}"
    BEST="$WORK/runs/$RUN_NAME/weights/best.pt"
    if [[ ! -f "$BEST" ]]; then
        echo "    [$MODEL] no best.pt — skipping inference"
        continue
    fi
    P4_JSON="$WORK/output/phase4_${MODEL}.json"
    CSV_DIR="$WORK/output/csv_${MODEL}"

    echo ""
    echo "    --- [$MODEL] Phase 4 ---"
    python "$REPO_ROOT/scripts/05_inference_ocr.py" \
        --weights "$BEST" \
        --source "$TEST_DIR" \
        --out-json "$P4_JSON" >/dev/null

    echo "    --- [$MODEL] Phase 5 ---"
    python "$REPO_ROOT/scripts/06_extract_points.py" \
        --phase4-json "$P4_JSON" \
        --out-dir "$CSV_DIR" \
        --meta-dir "$WORK/dataset/meta" >/dev/null

    echo "    [$MODEL] done — csv_${MODEL}/, phase4_${MODEL}.json"
done

# ============================================================
# [5/5] Stage everything important into output/ for one-folder download
# ============================================================
echo ""
echo "==> [5/5] Staging artifacts into output/"

# Per-model: copy best.pt + key training plots/CSVs out of runs/ into output/
# so the user only has to rsync output/ and gets everything that matters.
for MODEL in $MODELS; do
    RUN_NAME="${RUN_PREFIX}_${MODEL}"
    RUN_DIR="$WORK/runs/$RUN_NAME"
    BEST="$RUN_DIR/weights/best.pt"
    if [[ ! -f "$BEST" ]]; then
        echo "    [$MODEL] no best.pt — skipping stage"
        continue
    fi
    # Flat best-weights with clear names
    cp "$BEST" "$WORK/output/best_${MODEL}.pt"
    # Per-model training summary dir
    DEST="$WORK/output/train_${MODEL}"
    mkdir -p "$DEST"
    for f in results.png results.csv \
             labels.jpg labels_correlogram.jpg \
             confusion_matrix.png confusion_matrix_normalized.png \
             F1_curve.png P_curve.png R_curve.png PR_curve.png \
             val_batch0_labels.jpg val_batch0_pred.jpg \
             train_batch0.jpg train_batch1.jpg train_batch2.jpg \
             args.yaml; do
        [[ -f "$RUN_DIR/$f" ]] && cp "$RUN_DIR/$f" "$DEST/"
    done
done

# Aggregate comparison.csv at the root of output/
python "$REPO_ROOT/scripts/07_compare_models.py" \
    --runs   "$WORK/runs" \
    --output "$WORK/output" \
    --prefix "$RUN_PREFIX" \
    --out    "$WORK/output/comparison.csv"

# Drop a README inside output/ so the user knows what they're looking at after rsync.
cat > "$WORK/output/README.txt" <<EOF
Graph Digitizer — per-model results bundle
Generated $(date)

Files:
  comparison.csv          Per-model val mAP + test-extraction error medians.
  best_<MODEL>.pt         Best YOLO weights for that model.
  phase4_<MODEL>.json     Per-test-image detection + axis-OCR result.
  csv_<MODEL>/            Per-test-image extracted points (one CSV per chart)
                          and a summary.json with per-image accuracy.
  train_<MODEL>/          Training plots + results.csv + sample val batch
                          predictions + args.yaml (the hyperparams that ran).

Models in this run:
$(for M in $MODELS; do echo "  $M"; done)

Dataset: $N_TRAIN train + $N_VAL val + $N_TEST test, seed $SEED.
Budget : $BUDGET.
EOF

echo ""
echo "==========================================================="
echo "  DONE."
echo "==========================================================="
echo ""
echo "Comparison summary:"
column -s, -t < "$WORK/output/comparison.csv" || cat "$WORK/output/comparison.csv"
echo ""
echo "All download-worthy artifacts are in: $WORK/output/"
ls -la "$WORK/output/"
echo ""
echo "Pull back to your laptop (only this folder is needed):"
echo "  rsync -avz -e 'ssh -p <vast-port>' \\"
echo "    root@<vast-host>:$WORK/output/ \\"
echo "    \"\$HOME/Downloads/AI Project/output_vastai/\""
