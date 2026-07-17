#!/bin/bash
# run_single_model_evaluations.sh (Unified Version)
#
# Use the MODEL_TYPE variable to select either CLIP or LLaVA for evaluation.
#
# MODEL_TYPE options:
#   clip   - OpenAI CLIP / NegCLIP / ConCLIP / SigLIP, etc.
#   llava  - LLaVA-1.5 / LLaVA-NeXT (HuggingFace local checkpoints)

# Select model type
MODEL_TYPE="clip"   # Change to "clip" or "llava"

# Set the base directory for data and logs. Users should update this to their directory structure.
BASE_DIR="/content/negbench"  # Change this to your base directory
DATA_DIR="$BASE_DIR/benchmarks/data"  # Change this to your data directory
LOGS_DIR="$BASE_DIR/logs"
MODELS_DIR="$BASE_DIR/benchmarks/models"  # Change this to your models directory

# CLIP settings (used when MODEL_TYPE=clip)
MODEL="ViT-B-32"
MODEL_NAME="ViT_B_32_openai"
PRETRAINED_MODEL="openai"
# NegCLIP example:
#   MODEL_NAME="NegCLIP"
#   PRETRAINED_MODEL="$MODELS_DIR/NegCLIP/negclip.pth"
# ConCLIP example:
#   MODEL_NAME="ConCLIP"
#   PRETRAINED_MODEL="$MODELS_DIR/ConCLIP/conclip_b32_openclip_version.pt"

# LLaVA settings (used when MODEL_TYPE=llava)
# Path to the downloaded LLaVA checkpoint from HuggingFace
LLAVA_MODEL_PATH="$MODELS_DIR/llava-1.5-7b-hf"
LLAVA_MODEL_NAME="llava_1.5_7b"
# Vision encoder swap (optional):
#   Replaces the LLaVA vision tower with a fine-tuned CLIP model like NegCLIP.
#   Leave these variables empty if you do not want to swap the encoder (default).
VISION_ENCODER_MODEL=""   # Example: "ViT-B-32"
VISION_ENCODER_PATH=""    # Example: "$MODELS_DIR/NegCLIP/negclip.pth"
DTYPE="float16"           # float16 | bfloat16 | float32
MAX_NEW_TOKENS=16
# Quantization to reduce VRAM usage (requires: pip install bitsandbytes)
#   int4  : ~4-5 GB VRAM  (recommended for GPUs with 8-12 GB)
#   int8  : ~7-8 GB VRAM
#   ""    : no quantization (requires ~14 GB for LLaVA-1.5-7b in float16)
QUANTIZE="int4"

# Dataset paths for images
COCO_MCQ="$DATA_DIR/images/COCO_val_mcq_llama3.1_rephrased.csv"
VOC_MCQ="$DATA_DIR/images/VOC2007_mcq_llama3.1_rephrased.csv"
COCO_RETRIEVAL="$DATA_DIR/images/COCO_val_retrieval.csv"
COCO_NEGATED_RETRIEVAL="$DATA_DIR/images/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"

# Dataset paths for videos
MSRVTT_RETRIEVAL="$DATA_DIR/videos/MSRVTT/msr_vtt_retrieval.csv"
MSRVTT_NEGATED_RETRIEVAL="$DATA_DIR/videos/MSRVTT/negation/msr_vtt_retrieval_rephrased_llama.csv"
MSRVTT_MCQ="$DATA_DIR/videos/MSRVTT/negation/msr_vtt_mcq_rephrased_llama.csv"

# Activate the appropriate environment
source activate clip_negation || conda activate clip_negation

# Set system limits
ulimit -S -n 100000

# Logs directory for this evaluation run
RUN_LOGS_DIR="$LOGS_DIR/evaluation"
mkdir -p "$RUN_LOGS_DIR"

cd "$BASE_DIR/benchmarks/"

# Branch execution based on the model type
if [ "$MODEL_TYPE" = "clip" ]; then
    # Evaluate CLIP models (eval_negation.py)
    # Supports both MCQ and Retrieval
    echo "Starting CLIP Evaluation (MODEL_TYPE=clip)"
    echo "Model: $MODEL"
    echo "Pretrained: $PRETRAINED_MODEL"
    echo "Logs: $RUN_LOGS_DIR"

    CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
        --model "$MODEL" \
        --pretrained "$PRETRAINED_MODEL" \
        --name "image_${MODEL_NAME}" \
        --logs="$RUN_LOGS_DIR" \
        --dataset-type csv \
        --csv-separator=, \
        --csv-img-key filepath \
        --csv-caption-key caption \
        --zeroshot-frequency 1 \
        --coco-mcq="$COCO_MCQ" \
        --coco-retrieval="$COCO_RETRIEVAL" \
        --coco-negated-retrieval="$COCO_NEGATED_RETRIEVAL" \
        --batch-size=64 \
        --workers=8

elif [ "$MODEL_TYPE" = "llava" ]; then
    # Evaluate LLaVA models (eval_negation_llava.py)
    # Supports MCQ only (Retrieval is not supported)
    # --shuffle-mcq-options is strongly recommended to remove position bias
    echo "Starting LLaVA MCQ Evaluation (MODEL_TYPE=llava)"
    echo "Model: $LLAVA_MODEL_PATH"
    echo "Encoder: ${VISION_ENCODER_PATH:-original (no swap)}"
    echo "Logs: $RUN_LOGS_DIR"

    # Configure flags for vision encoder swapping
    ENCODER_FLAGS=""
    if [ -n "$VISION_ENCODER_PATH" ] && [ -n "$VISION_ENCODER_MODEL" ]; then
        ENCODER_FLAGS="--vision-encoder-model $VISION_ENCODER_MODEL \
                       --vision-encoder-path $VISION_ENCODER_PATH"
        LLAVA_MODEL_NAME="${LLAVA_MODEL_NAME}_enc_${VISION_ENCODER_MODEL}"
    fi

    # Configure flag for quantization
    QUANTIZE_FLAG=""
    if [ -n "$QUANTIZE" ]; then
        QUANTIZE_FLAG="--quantize $QUANTIZE"
    fi

    CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation_llava \
        --llava-model-path "$LLAVA_MODEL_PATH" \
        $ENCODER_FLAGS \
        $QUANTIZE_FLAG \
        --name "mcq_${LLAVA_MODEL_NAME}" \
        --logs="$RUN_LOGS_DIR" \
        --coco-mcq="$COCO_MCQ" \
        --voc2007-mcq="$VOC_MCQ" \
        --shuffle-mcq-options \
        --seed 42 \
        --device cuda \
        --dtype "$DTYPE" \
        --max-new-tokens "$MAX_NEW_TOKENS"

else
    echo "Error: Unknown MODEL_TYPE='$MODEL_TYPE'." >&2
    echo "       Please use 'clip' or 'llava'." >&2
    exit 1
fi

echo ""
echo "Evaluation complete. Logs saved to $RUN_LOGS_DIR."
