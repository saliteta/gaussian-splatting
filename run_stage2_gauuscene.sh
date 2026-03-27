#!/usr/bin/env bash
# =============================================================================
# run_stage2_gauuscene.sh
#
# For each of the 7 GauUscene scenes:
#   1. Convert cloud_merged.las → voxel-downsampled PLY (0.08 m)
#      and place it at colmap_metrics/sparse/0/points3D.ply
#      (original COLMAP PLY is backed up as points3D_colmap_backup.ply)
#   2. Run train_stage2.py directly from that point cloud
#
#   Iterations  = num_images_in_scene × 200
#   Save milestones = 5 evenly-spaced checkpoints across total iterations
#
# Usage:
#   bash run_stage2_gauuscene.sh [SCENE1 SCENE2 ...]   # run specific scenes
#   bash run_stage2_gauuscene.sh                        # run all 7 scenes
# =============================================================================
set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_ROOT="/mnt/z/Dataset/GauUscene/GauUsceneDepth"
OUTPUT_ROOT="/mnt/z/Users/butian/GauUscene"
REPO_DIR="/home/saliteta/workspace/gaussian-splatting"
CONDA_ENV="GauUscene"

# ── Stage-2 hyper-parameters (fixed across scenes) ───────────────────────────
VOXEL_SIZE="0.08"       # used for LAS downsample AND spatial block index
ITERS_PER_IMAGE=200     # iterations = num_images × this value
N_SAVES=5               # number of evenly-spaced save milestones
BATCH_SIZE="8"
SLOT_BUDGET_GB="2.0"
IOU_SAMPLE="500000"
FOV_MARGIN="0.1"

# ── Scene list ────────────────────────────────────────────────────────────────
ALL_SCENES=(CUHK_LOWER CUHK_UPPER HAV LFLS SMBU SZIIT SZTU)

if [ $# -gt 0 ]; then
    SCENES=("$@")
else
    SCENES=("${ALL_SCENES[@]}")
fi

# ── Helper: log with timestamp ────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Helper: build evenly-spaced milestone string ──────────────────────────────
# milestones N_SAVES <total>  →  "step1 step2 ... total"
milestones() {
    local n=$1
    local total=$2
    local out=""
    for i in $(seq 1 "$n"); do
        local ms=$(( total * i / n ))
        out="$out $ms"
    done
    echo "$out"
}

# =============================================================================
mkdir -p "$OUTPUT_ROOT"

for SCENE in "${SCENES[@]}"; do
    log "════════════════════════════════════════════"
    log "Scene: $SCENE"
    log "════════════════════════════════════════════"

    SCENE_DIR="$DATASET_ROOT/$SCENE"
    COLMAP_DIR="$SCENE_DIR/colmap_metrics"
    LAS_FILE="$SCENE_DIR/cloud_merged.las"
    SPARSE_DIR="$COLMAP_DIR/sparse/0"
    PLY_OUT="$SPARSE_DIR/points3D.ply"
    PLY_BACKUP="$SPARSE_DIR/points3D_colmap_backup.ply"
    MODEL_OUT="$OUTPUT_ROOT/$SCENE"

    # ── Sanity checks ─────────────────────────────────────────────────────────
    if [ ! -f "$LAS_FILE" ]; then
        log "ERROR: $LAS_FILE not found — skipping $SCENE"; continue
    fi
    if [ ! -d "$SPARSE_DIR" ]; then
        log "ERROR: $SPARSE_DIR not found — skipping $SCENE"; continue
    fi
    if [ ! -d "$COLMAP_DIR/images" ]; then
        log "ERROR: $COLMAP_DIR/images not found — skipping $SCENE"; continue
    fi

    # ── Compute per-scene iterations and milestones ───────────────────────────
    NUM_IMAGES=$(ls "$COLMAP_DIR/images" | wc -l)
    ITERATIONS=$(( NUM_IMAGES * ITERS_PER_IMAGE ))
    SAVE_ITERS=$(milestones "$N_SAVES" "$ITERATIONS")
    log "Images: $NUM_IMAGES  →  iterations: $ITERATIONS"
    log "Save milestones:$SAVE_ITERS"

    # ── Step 1: backup original COLMAP PLY ────────────────────────────────────
    if [ -f "$PLY_OUT" ] && [ ! -f "$PLY_BACKUP" ]; then
        log "Backing up original COLMAP PLY → points3D_colmap_backup.ply"
        cp "$PLY_OUT" "$PLY_BACKUP"
    fi

    # ── Step 2: LAS → voxel-downsampled PLY ──────────────────────────────────
    log "Converting cloud_merged.las → PLY (voxel_size=${VOXEL_SIZE} m) ..."
    conda run -n "$CONDA_ENV" python3 "$REPO_DIR/las_to_ply.py" \
        "$LAS_FILE" \
        "$PLY_OUT" \
        "$VOXEL_SIZE"

    # ── Step 3: Stage-2 training ──────────────────────────────────────────────
    log "Starting Stage-2 training → $MODEL_OUT"
    mkdir -p "$MODEL_OUT"

    LOG_FILE="$MODEL_OUT/train_stage2.log"
    log "Log file: $LOG_FILE"

    conda run -n "$CONDA_ENV" python3 "$REPO_DIR/train_stage2.py" \
        --source_path    "$COLMAP_DIR"     \
        --model_path     "$MODEL_OUT"      \
        --images         images            \
        --iterations     "$ITERATIONS"     \
        --batch_size     "$BATCH_SIZE"     \
        --slot_budget_gb "$SLOT_BUDGET_GB" \
        --voxel_size     "$VOXEL_SIZE"     \
        --iou_sample     "$IOU_SAMPLE"     \
        --fov_margin     "$FOV_MARGIN"     \
        --save_iterations $SAVE_ITERS \
        2>&1 | tee "$LOG_FILE"

    log "Done: $SCENE  →  $MODEL_OUT"
    echo ""
done

log "All scenes complete."
