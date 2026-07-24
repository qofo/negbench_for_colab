#!/bin/bash

# ==============================================================================
# Script: run_pca.sh
# Purpose: Run CLIP Layer-wise PCA Analysis on COCO_val_mcq_top100_uncovered.csv
# Output Directory: results/pca/coco_val_mcq_top100_uncovered/
# ==============================================================================

# 입력 데이터셋 경로
CSV_PATH="benchmarks/data/images/COCO_val_mcq_top100_uncovered.csv"

# 결과 출력 디렉토리
OUTPUT_DIR="logs/pca/coco_val_mcq_top100_uncovered"

# 모델 설정
MODEL_NAME="ViT-B-32"
PRETRAINED="openai"
TARGET_TOKEN="eot"  # 옵션: 'eot' (기본값), 'mean', 'all'
MAX_SAMPLES=500     # 사용할 샘플 문장 쌍 개수

echo "=========================================================="
echo "Starting CLIP Layer-wise PCA Analysis"
echo "Input CSV   : ${CSV_PATH}"
echo "Output Dir  : ${OUTPUT_DIR}"
echo "Model       : ${MODEL_NAME} (${PRETRAINED})"
echo "Target Token: ${TARGET_TOKEN}"
echo "=========================================================="

python benchmarks/src/evaluation/pca_text_encoder.py \
    --csv_path "${CSV_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --model "${MODEL_NAME}" \
    --pretrained "${PRETRAINED}" \
    --target_token "${TARGET_TOKEN}" \
    --max_samples ${MAX_SAMPLES}

echo "=========================================================="
echo "PCA Analysis Complete!"
echo "Saved PCA grid plot to: ${OUTPUT_DIR}/clip_layer_pca_${TARGET_TOKEN}.png"
echo "Saved text report to  : ${OUTPUT_DIR}/pca_analysis_report_${TARGET_TOKEN}.txt"
echo "=========================================================="
