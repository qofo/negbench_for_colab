#!/bin/bash
# ============================================================
# run_single_model_evaluations_llava.sh
#
# Evaluate a local LLaVA model on NegBench MCQ tasks.
# Mirrors the interface of run_single_model_evaluations.sh.
#
# Usage:
#   bash scripts/run_single_model_evaluations_llava.sh
#
# ---- Default: evaluate LLaVA with its original vision tower ----
# Set VISION_ENCODER_PATH to "" to skip the encoder swap.
#
# ---- Swap the vision encoder ----
# Set VISION_ENCODER_MODEL and VISION_ENCODER_PATH to use
# a fine-tuned CLIP (e.g., NegCLIP or CC12M-NegFull) as the
# visual backbone instead of LLaVA's built-in SigLIP/CLIP.
# ============================================================

# ----- Base directories (update these to match your setup) -----
BASE_DIR="/content/negbench"
DATA_DIR="$BASE_DIR/benchmarks/data"
LOGS_DIR="$BASE_DIR/logs"
MODELS_DIR="$BASE_DIR/benchmarks/models"

# ----- LLaVA model path -----
# Point this to the local directory of your downloaded LLaVA checkpoint.
# Example: llava-hf/llava-1.5-7b-hf downloaded via huggingface-cli
LLAVA_MODEL_PATH="$MODELS_DIR/llava-1.5-7b-hf"

# Experiment name (used for log directory)
MODEL_NAME="llava_1.5_7b"

# ----- Optional: vision encoder swap -----
# Leave VISION_ENCODER_PATH empty ("") to keep LLaVA's original tower.
# Set both to replace it with an OpenCLIP-compatible checkpoint.
#
# Example 1 – NegCLIP:
#   VISION_ENCODER_MODEL="ViT-B-32"
#   VISION_ENCODER_PATH="$MODELS_DIR/NegCLIP/negclip.pth"
#
# Example 2 – Our fine-tuned CC12M-NegFull model:
#   VISION_ENCODER_MODEL="ViT-B-32"
#   VISION_ENCODER_PATH="$MODELS_DIR/NegCLIP_CC12M_NegFull_ViT-B-32_lr1e-8_clw0.99_mlw0.01/finetuned_checkpoint.pt"
VISION_ENCODER_MODEL=""       # e.g. "ViT-B-32"
VISION_ENCODER_PATH=""        # e.g. "$MODELS_DIR/NegCLIP/negclip.pth"

# ----- Dataset paths -----
COCO_MCQ="$DATA_DIR/images/COCO_val_mcq_llama3.1_rephrased.csv"
VOC_MCQ="$DATA_DIR/images/VOC2007_mcq_llama3.1_rephrased.csv"

# ----- Settings -----
SEED=42
DEVICE="cuda"
DTYPE="float16"       # float16 | bfloat16 | float32
MAX_NEW_TOKENS=16

# ----- Environment -----
source activate clip_negation || conda activate clip_negation
ulimit -S -n 100000

RUN_LOGS_DIR="$LOGS_DIR/evaluation"
mkdir -p "$RUN_LOGS_DIR"

cd "$BASE_DIR/benchmarks/"

# ----- Build optional encoder swap flags -----
ENCODER_FLAGS=""
if [ -n "$VISION_ENCODER_PATH" ] && [ -n "$VISION_ENCODER_MODEL" ]; then
    ENCODER_FLAGS="--vision-encoder-model $VISION_ENCODER_MODEL \
                   --vision-encoder-path $VISION_ENCODER_PATH"
    MODEL_NAME="${MODEL_NAME}_enc_${VISION_ENCODER_MODEL}"
fi

echo "Starting LLaVA MCQ Evaluation..."
echo "  Model:   $LLAVA_MODEL_PATH"
echo "  Encoder: ${VISION_ENCODER_PATH:-original}"
echo "  Logs:    $RUN_LOGS_DIR"

CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation_llava \
    --llava-model-path "$LLAVA_MODEL_PATH" \
    $ENCODER_FLAGS \
    --name "mcq_${MODEL_NAME}" \
    --logs="$RUN_LOGS_DIR" \
    --coco-mcq="$COCO_MCQ" \
    --voc2007-mcq="$VOC_MCQ" \
    --shuffle-mcq-options \
    --seed $SEED \
    --device $DEVICE \
    --dtype $DTYPE \
    --max-new-tokens $MAX_NEW_TOKENS

echo "LLaVA evaluation complete. Logs saved to $RUN_LOGS_DIR."
