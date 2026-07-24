#!/bin/bash

# ==============================================================================
# Script: run_pipeline_breakdown.sh (Refined 3rd Edition)
# Experiments:
#   Stage 1-A: Pipeline 5-Step Degradation Line Plot
#   Stage 1-B: Direction Preservation Analysis (Distance Ratio: Negation vs Control)
#   Stage 1-C: Linear Probe Analysis (LogisticRegression Acc: Pre vs Post Projection)
#   Stage 1-D: PCA Variance Spectrum Compression (Pre vs Post Projection Geometry)
#   Stage 2  : Image-Text Retrieval Metrics (Binary Acc, Flip Rate - if image_root set)
#   Stage 3  : Text-side Projection Causal Ablation (Original vs Identity vs Orthogonal Q)
# ==============================================================================

PAIRED_CSV="COCO_val_full_paired.csv"
if [ ! -f "${PAIRED_CSV}" ]; then
    PAIRED_CSV="COCO_val_mcq_top100_paired.csv"
fi

OUTPUT_DIR="logs/pipeline_breakdown/openai_vit_b32"
MODEL_NAME="ViT-B-32"
PRETRAINED="openai"
TARGET_TOKEN="eot"
MAX_SAMPLES=60000
BATCH_SIZE=256

# 이미지 저장 서버인 경우 경로 지정 (예: IMAGE_ROOT=".")
IMAGE_ROOT=""

echo "=========================================================="
echo "  Executing Stage-by-Stage CLIP Negation Analysis"
echo "  Input CSV   : ${PAIRED_CSV}"
echo "  Output Dir  : ${OUTPUT_DIR}"
echo "  Model       : ${MODEL_NAME} (${PRETRAINED})"
echo "  Batch Size  : ${BATCH_SIZE}"
echo "  Image Root  : $([ -z \"${IMAGE_ROOT}\" ] && echo 'SKIP (no image_root)' || echo ${IMAGE_ROOT})"
echo "=========================================================="

CMD="python benchmarks/src/evaluation/pca_text_encoder.py \
    --csv_path ${PAIRED_CSV} \
    --output_dir ${OUTPUT_DIR} \
    --model ${MODEL_NAME} \
    --pretrained ${PRETRAINED} \
    --target_token ${TARGET_TOKEN} \
    --max_samples ${MAX_SAMPLES} \
    --batch_size ${BATCH_SIZE}"

if [ -n "${IMAGE_ROOT}" ]; then
    CMD="${CMD} --image_root ${IMAGE_ROOT}"
fi

eval ${CMD}

echo "=========================================================="
echo "Analysis Complete! Generated experimental outputs:"
echo "  [Stage 1-A] ${OUTPUT_DIR}/pipeline_step_lineplot.png"
echo "  [Stage 1-B] ${OUTPUT_DIR}/direction_preservation_analysis.png"
echo "  [Stage 1-C] ${OUTPUT_DIR}/linear_probe_accuracy.png        (Linear Separability)"
echo "  [Stage 1-D] ${OUTPUT_DIR}/pca_spectrum_compression.png      (PCA Variance Spectrum)"
echo "  [Stage 1-D] ${OUTPUT_DIR}/pca_spectrum_report.json"
echo "  [Stage 3]   ${OUTPUT_DIR}/projection_causal_ablation.png"
if [ -n "${IMAGE_ROOT}" ]; then
echo "  [Stage 2]   ${OUTPUT_DIR}/retrieval_metrics_summary.json"
fi
echo "=========================================================="
