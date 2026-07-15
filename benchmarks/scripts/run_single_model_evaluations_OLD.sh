#!/bin/bash

# Set the base directory for data and logs. Users should update this to their directory structure.
BASE_DIR="/path/to/your/research/project"  # Change this to your base directory
DATA_DIR="path/to/your/data"  # Change this to your data directory
LOGS_DIR="$BASE_DIR/logs"
MODELS_DIR="path/to/your/models"  # Change this to your models directory

# Model and pretrained options
MODEL="ViT-B-32"
MODEL_NAME="NegCLIP"
PRETRAINED_MODEL="$MODELS_DIR/$MODEL_NAME/name_of_your_model.pt" # Change this to the real path of your model
# Note: for an openclip model, you can use the following:
# MODEL_NAME="ViT_B_32_openai"
# PRETRAINED_MODEL="openai"

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

cd ..

# Image Evaluation
echo "Starting Image Evaluation..."
# Note: you can add --report-to wandb to report to wandb
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
    --model $MODEL \
    --pretrained $PRETRAINED_MODEL \
    --name "image_$MODEL_NAME" \
    --logs=$RUN_LOGS_DIR \
    --dataset-type csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --zeroshot-frequency 1 \
    --imagenet-val="$DATA_DIR/images/imagenet" \
    --coco-mcq=$COCO_MCQ \
    --voc2007-mcq=$VOC_MCQ \
    --coco-retrieval=$COCO_RETRIEVAL \
    --coco-negated-retrieval=$COCO_NEGATED_RETRIEVAL \
    --batch-size=64 \
    --workers=8

# Video Evaluation
echo "Starting Video Evaluation..."
# Note: you can add --report-to wandb to report to wandb
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
    --model $MODEL \
    --pretrained $PRETRAINED_MODEL \
    --name "video_$MODEL_NAME" \
    --logs=$RUN_LOGS_DIR \
    --dataset-type csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --zeroshot-frequency 1 \
    --imagenet-val="$DATA_DIR/images/imagenet" \
    --msrvtt-retrieval=$MSRVTT_RETRIEVAL \
    --msrvtt-negated-retrieval=$MSRVTT_NEGATED_RETRIEVAL \
    --msrvtt-mcq=$MSRVTT_MCQ \
    --video \
    --batch-size=64 \
    --workers=8

echo "Evaluation complete. Logs saved to $RUN_LOGS_DIR."
