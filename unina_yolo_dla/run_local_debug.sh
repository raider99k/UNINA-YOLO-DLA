#!/bin/bash
# =============================================================================
# UNINA-YOLO-DLA: Local Debug Dry-Run Script
# =============================================================================
#
# This script orchestrates a complete dry-run of the training pipeline
# on CPU-only systems (no CUDA, no TensorRT).
#
# What it does:
#   1. Installs base dependencies (if needed)
#   2. Generates a synthetic FSD dataset
#   3. Runs train.py with minimal epochs for logic verification
#
# Usage:
#   bash run_local_debug.sh
#
# Options:
#   --skip-deps    Skip dependency installation
#   --skip-gen     Skip dataset generation (use existing)
#   --epochs N     Number of training epochs (default: 2)
#
# =============================================================================

set -e  # Exit on error

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPOCHS=2
BATCH_SIZE=4
IMG_SIZE=320  # Smaller for faster testing
SKIP_DEPS=false
SKIP_GEN=false

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --skip-gen)
            SKIP_GEN=true
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "UNINA-YOLO-DLA: Local Debug Dry-Run"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Epochs:     $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Image Size: $IMG_SIZE"
echo "  Skip Deps:  $SKIP_DEPS"
echo "  Skip Gen:   $SKIP_GEN"
echo ""

# --- Step 1: Install Dependencies ---
if [ "$SKIP_DEPS" = false ]; then
    echo ">>> Step 1: Installing dependencies..."
    pip install --quiet --upgrade pip
    pip install --quiet ultralytics opencv-python numpy torch
    echo "    Dependencies installed."
else
    echo ">>> Step 1: Skipping dependency installation."
fi
echo ""

# --- Step 2: Generate Synthetic Dataset ---
if [ "$SKIP_GEN" = false ]; then
    echo ">>> Step 2: Generating synthetic FSD dataset..."
    python "$SCRIPT_DIR/tools/generate_synthetic_fsd.py" \
        --num-train 50 \
        --num-val 10 \
        --seed 42
    echo "    Dataset generated."
else
    echo ">>> Step 2: Skipping dataset generation."
fi
echo ""

# --- Step 3: Verify Dataset Structure ---
echo ">>> Step 3: Verifying dataset structure..."
DATASET_DIR="$SCRIPT_DIR/datasets/fsd_synth"
CONFIG_FILE="$DATASET_DIR/fsd_synth.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Dataset config not found at $CONFIG_FILE"
    exit 1
fi

TRAIN_IMAGES=$(ls -1 "$DATASET_DIR/images/train" 2>/dev/null | wc -l)
VAL_IMAGES=$(ls -1 "$DATASET_DIR/images/val" 2>/dev/null | wc -l)
echo "    Train images: $TRAIN_IMAGES"
echo "    Val images:   $VAL_IMAGES"

if [ "$TRAIN_IMAGES" -lt 10 ] || [ "$VAL_IMAGES" -lt 5 ]; then
    echo "ERROR: Insufficient dataset images."
    exit 1
fi
echo "    Dataset structure OK."
echo ""

# --- Step 4: Run Training Dry-Run ---
echo ">>> Step 4: Running training dry-run (CPU mode)..."
echo "    This verifies the training loop completes without errors."
echo ""

# Set environment variable to force mocks if CUDA not available
export UNINA_FORCE_MOCKS=1

python "$SCRIPT_DIR/train.py" \
    --data "$CONFIG_FILE" \
    --epochs "$EPOCHS" \
    --batch "$BATCH_SIZE" \
    --imgsz "$IMG_SIZE" \
    --device cpu \
    --project runs/local_debug \
    --name dry_run \
    --exist-ok

echo ""
echo "============================================================"
echo ">>> DRY-RUN COMPLETE!"
echo "============================================================"
echo ""
echo "Results:"
echo "  - Training loop completed without errors"
echo "  - Output saved to: runs/local_debug/dry_run"
echo ""
echo "Next steps:"
echo "  1. Review the training logs for any warnings"
echo "  2. Check that custom metrics (mAP_small) are being computed"
echo "  3. For full training, use real FSD dataset on GPU"
echo ""
