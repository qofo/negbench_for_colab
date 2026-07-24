"""
reprocess_llava_predictions.py

Post-processes a LLaVA MCQ predictions CSV produced by eval_negation_llava.py.
Fixes three issues found in the raw output:

  1. PARSING BUG: The default parser uses a plain substring search, which always
     finds the letter 'A' in almost any English sentence (e.g. "Answer", "A cat",
     etc.). This makes the model appear to always predict option A regardless of
     what it actually generated.  This script replaces that logic with robust
     pattern-based parsing that handles outputs like "(B)", "The answer is C",
     "Option D", etc.

  2. SHUFFLED ORDER: The predictions CSV stores captions in the shuffled order
     that was presented to the model, not in the original dataset order.
     This script restores the original order (caption_0 = ground-truth answer,
     caption_1 = hybrid, caption_2 = positive, caption_3 = negative) and adjusts
     correct_answer / predicted_answer indices accordingly.

  3. LOGITS (optional): When --llava-model-path is supplied the script re-runs
     the model with output_scores=True and extracts the probability of each
     option letter (A/B/C/D) from the first generated token.  The probabilities
     are stored as logit_0 .. logit_{n-1} in the original caption order.

Usage
    Fast mode (no model, fixes 1 & 2 only):
        python -m src.evaluation.reprocess_llava_predictions \\
            --input-csv  logs/evaluation/mcq_llava/predictions/coco-mcq_predictions.csv \\
            --dataset-csv data/images/COCO_val_mcq_llama3.1_rephrased.csv \\
            --output-csv  logs/evaluation/mcq_llava/predictions/coco-mcq_reprocessed.csv

    Full mode (fixes 1, 2 & 3):
        python -m src.evaluation.reprocess_llava_predictions \\
            --input-csv  logs/evaluation/mcq_llava/predictions/coco-mcq_predictions.csv \\
            --dataset-csv data/images/COCO_val_mcq_llama3.1_rephrased.csv \\
            --output-csv  logs/evaluation/mcq_llava/predictions/coco-mcq_reprocessed.csv \\
            --llava-model-path /path/to/llava-1.5-7b-hf \\
            --device cuda \\
            --dtype float16 \\
            --quantize int4
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.llava.metrics import compute_mcq_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

from src.llava.parser import OPTION_LABELS, parse_option_robust
from src.llava.dataset_utils import build_dataset_index, restore_original_order, build_perm
from src.llava.logits import get_option_token_ids, extract_option_logits


# Main pipeline

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Reprocess LLaVA MCQ predictions: fix parsing, restore original order, optionally extract logits."
    )
    parser.add_argument(
        "--input-csv", required=True,
        help="Path to the existing predictions CSV (output of eval_negation_llava.py).",
    )
    parser.add_argument(
        "--dataset-csv", required=True,
        help="Path to the original NegBench MCQ dataset CSV (used to restore original caption order).",
    )
    parser.add_argument(
        "--output-csv", required=True,
        help="Path for the reprocessed output CSV.",
    )
    parser.add_argument(
        "--llava-model-path", default=None,
        help="Path to a LLaVA checkpoint.  Required only for logit extraction. "
             "If omitted, logit columns are filled with NaN.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", default="float16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument(
        "--quantize", default=None, choices=["int4", "int8"],
        help="BitsAndBytes quantization (requires bitsandbytes). "
             "Use int4 for GPUs with less than 16 GB VRAM.",
    )
    return parser.parse_args(args)




def main(args):
    args = parse_args(args)

    logger.info("Loading input CSV: %s", args.input_csv)
    pred_df = pd.read_csv(args.input_csv)

    logger.info("Loading dataset CSV: %s", args.dataset_csv)
    ds_df = build_dataset_index(args.dataset_csv)

    cap_cols_pred = sorted(
        [c for c in pred_df.columns if re.fullmatch(r"caption_\d+", c)],
        key=lambda c: int(c.split("_")[1]),
    )
    n_options = len(cap_cols_pred)
    labels = OPTION_LABELS[:n_options]

    # Optionally load model for logit extraction
    model = processor = option_token_ids = None
    use_device_map = False
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    if args.llava_model_path is not None:
        logger.info("Loading LLaVA model for logit extraction from %s ...", args.llava_model_path)
        from src.llava.llava_evaluator import LLaVAModularEvaluator
        evaluator = LLaVAModularEvaluator(
            model_path=args.llava_model_path,
            device=device,
            dtype=dtype,
            quantize=args.quantize
        )
        model = evaluator._full_model
        processor = evaluator.processor
        use_device_map = getattr(evaluator, "_use_device_map", False)
        option_token_ids = get_option_token_ids(processor.tokenizer, labels)

    records = []

    for _, row in tqdm(pred_df.iterrows(), total=len(pred_df), desc="Reprocessing"):
        image_path = str(row["image_path"])
        shuffled_captions = [str(row[c]) for c in cap_cols_pred]
        shuffled_correct = int(row["correct_answer"])
        raw_generated = str(row.get("raw_generated", ""))
        question_type = str(row.get("question_type", ""))

        # Fix 1: robust parsing of raw_generated
        robust_predicted_shuffled = parse_option_robust(raw_generated, labels)

        # Fix 2: restore original caption order
        orig_captions, orig_correct, orig_predicted, orig_types, matched = (
            restore_original_order(
                shuffled_captions,
                shuffled_correct,
                robust_predicted_shuffled,
                image_path,
                ds_df,
            )
        )
        is_correct = orig_predicted == orig_correct

        record: dict = {
            "image_path": image_path,
            "question_type": question_type,
            "correct_answer": orig_correct,
            "predicted_answer": orig_predicted,
            "is_correct": is_correct,
            "raw_generated": raw_generated,
            "caption_types": orig_types,
        }

        # Captions in original order
        for i, (cap, ctype) in enumerate(zip(orig_captions, orig_types)):
            record[f"caption_{i}"] = cap
            record[f"caption_type_{i}"] = ctype

        # Fix 3: logits (optional, requires model re-run)
        if model is not None:
            # Logit extraction uses the SHUFFLED captions (same as original evaluation)
            # but the resulting probabilities are re-ordered to original order
            try:
                shuffled_probs = extract_option_logits(
                    model, processor,
                    image_path, shuffled_captions,
                    labels, option_token_ids,
                    device, dtype, use_device_map,
                )
                # shuffled_probs[i] is the probability for the option shown at shuffled position i
                # We need to find which shuffled index corresponds to each original index
                perm = build_perm(shuffled_captions, orig_captions)
                # perm[orig_idx] = shuffled_idx
                orig_probs = [float(shuffled_probs[s_i]) for s_i in perm]
                for orig_i, prob in enumerate(orig_probs):
                    record[f"logit_{orig_i}"] = prob
                
                pred_logit_idx = int(np.argmax(orig_probs))
                record["predicted_answer_logit"] = pred_logit_idx
                record["is_correct_logit"] = (pred_logit_idx == record["correct_answer"])
            except Exception as exc:
                logger.warning("Logit extraction failed for %s: %s", image_path, exc)
                for i in range(n_options):
                    record[f"logit_{i}"] = float("nan")
        else:
            for i in range(n_options):
                record[f"logit_{i}"] = float("nan")

        records.append(record)

    out_df = pd.DataFrame(records)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    logger.info("Saved %d rows to %s", len(out_df), args.output_csv)

    # Print accuracy summary
    metrics = compute_mcq_metrics(records)
    logger.info("=" * 60)
    logger.info("Reprocessing complete. Summary:")
    for k, v in metrics.items():
        if k == "sample_results":
            continue
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        elif isinstance(v, dict):
            logger.info(f"  {k}: {v}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main(sys.argv[1:])
