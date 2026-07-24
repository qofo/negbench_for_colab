"""
LLaVA MCQ Evaluation Entry Point
==================================
Evaluates a LLaVA model on NegBench MCQ tasks (COCO, VOC2007, Synthetic).
Mirrors the interface of eval_negation.py so that LLaVA can be called
from a shell script in the same style as OpenCLIP / NegCLIP evaluations.

Key differences from CLIP evaluation
--------------------------------------
- Uses ``LLaVAModularEvaluator`` instead of ``create_model_and_transforms``.
- Images are fed one-by-one through the HF processor (no batch encoding).
- The vision encoder can be swapped to a fine-tuned CLIP checkpoint via
  ``--vision-encoder-path`` and ``--vision-encoder-model``.
- ``--shuffle-mcq-options`` is **strongly recommended** to remove position bias.

Example usage
-------------
    python -m src.evaluation.eval_negation_llava \\
        --llava-model-path /path/to/llava-1.5-7b-hf \\
        --name llava_coco_shuffled \\
        --logs /path/to/logs \\
        --coco-mcq /path/to/COCO_val_mcq_llama3.1_rephrased.csv \\
        --shuffle-mcq-options \\
        --seed 42

To swap the vision encoder with a fine-tuned CLIP:
    python -m src.evaluation.eval_negation_llava \\
        --llava-model-path /path/to/llava-1.5-7b-hf \\
        --vision-encoder-model ViT-B-32 \\
        --vision-encoder-path /path/to/negclip.pth \\
        --name llava_negclip_encoder \\
        --logs /path/to/logs \\
        --coco-mcq /path/to/COCO_val_mcq_llama3.1_rephrased.csv \\
        --shuffle-mcq-options
"""

import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.llava.llava_evaluator import LLaVAModularEvaluator
from src.llava.metrics import compute_mcq_metrics
from training.data import CsvMCQDataset


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Evaluate a LLaVA model on NegBench MCQ tasks."
    )

    # --- model ---
    parser.add_argument(
        "--llava-model-path", type=str, required=True,
        help="Path to the local LLaVA checkpoint directory."
    )
    parser.add_argument(
        "--vision-encoder-model", type=str, default=None,
        help="OpenCLIP architecture to use as replacement vision tower (e.g. ViT-B-32)."
    )
    parser.add_argument(
        "--vision-encoder-path", type=str, default=None,
        help="Path to an OpenCLIP checkpoint whose visual encoder replaces LLaVA's. "
             "Requires --vision-encoder-model."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="PyTorch device."
    )
    parser.add_argument(
        "--dtype", type=str, default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model weight dtype."
    )
    parser.add_argument(
        "--quantize", type=str, default=None,
        choices=["int4", "int8"],
        help="Enable BitsAndBytes quantization to reduce VRAM usage. "
             "int4 uses ~4-5 GB, int8 uses ~7-8 GB (requires bitsandbytes). "
             "Recommended for GPUs with less than 16 GB VRAM."
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=16,
        help="Max tokens generated per MCQ answer."
    )

    # --- evaluation data (mirrors eval_negation.py) ---
    parser.add_argument("--coco-mcq", type=str, default=None,
                        help="Path to COCO MCQ CSV.")
    parser.add_argument("--voc2007-mcq", type=str, default=None,
                        help="Path to VOC2007 MCQ CSV.")
    parser.add_argument("--synthetic-mcq", type=str, default=None,
                        help="Path to Synthetic MCQ CSV.")

    # --- shuffle ---
    parser.add_argument(
        "--shuffle-mcq-options", default=False, action="store_true",
        help="Shuffle answer order to eliminate position bias (strongly recommended)."
    )
    parser.add_argument("--seed", type=int, default=42)

    # --- logging ---
    parser.add_argument("--name", type=str, required=True,
                        help="Experiment name for log directory.")
    parser.add_argument("--logs", type=str, default="./logs/",
                        help="Directory to store results.")

    return parser.parse_args(args)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_llava_on_mcq(
    evaluator: LLaVAModularEvaluator,
    csv_path: str,
    shuffle_options: bool = True,
    seed: int = 42,
    dataset_name: str = "dataset",
) -> dict:
    """
    Run LLaVA MCQ evaluation on a single CSV file.

    Args:
        evaluator:       Loaded ``LLaVAModularEvaluator`` instance.
        csv_path:        Path to a NegBench MCQ CSV.
        shuffle_options: Whether to shuffle answer choices per sample.
        seed:            RNG seed for reproducible shuffling.
        dataset_name:    Short name used in metric keys.

    Returns:
        dict of evaluation metrics.
    """
    # Reuse CsvMCQDataset for consistent data loading + shuffling
    dataset = CsvMCQDataset(
        csv_file=csv_path,
        transforms=lambda x: x,   # PIL images are passed directly to LLaVA
        shuffle_options=shuffle_options,
        seed=seed,
    )

    sample_results = []

    for idx in tqdm(range(len(dataset)), desc=f"Evaluating {dataset_name}"):
        # Unpack the 6-tuple returned by the updated CsvMCQDataset
        image, captions, correct_answer, question_type, image_path, caption_types = dataset[idx]

        # captions is a list of str when tokenizer=None (default)
        assert isinstance(captions[0], str), \
            "captions should be strings; do not pass a tokenizer to CsvMCQDataset for LLaVA."

        # Load PIL image (dataset returns PIL when transforms=identity)
        pil_image = Image.open(image_path).convert("RGB")

        # Generate answer
        predicted_index, raw_text, option_probs = evaluator.generate_mcq_answer(pil_image, captions)

        is_correct = (predicted_index == correct_answer)

        sample_dict = {
            "image_path":       image_path,
            "question_type":    question_type,
            "correct_answer":   correct_answer,
            "predicted_answer": predicted_index,
            "is_correct":       is_correct,
            "raw_generated":    raw_text,
            "caption_types":    caption_types,
            **{f"caption_{j}": captions[j] for j in range(len(captions))},
        }
        if option_probs is not None:
            predicted_index_logit = int(np.argmax(option_probs))
            sample_dict["predicted_answer_logit"] = predicted_index_logit
            sample_dict["is_correct_logit"] = (predicted_index_logit == correct_answer)
            for j, prob in enumerate(option_probs):
                sample_dict[f"logit_{j}"] = float(prob)

        sample_results.append(sample_dict)

    return compute_mcq_metrics(sample_results, dataset_name=dataset_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    args = parse_args(args)

    # Logging
    log_dir = os.path.join(args.logs, args.name)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, "out.log")),
        ],
    )

    # Dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Load LLaVA
    evaluator = LLaVAModularEvaluator(
        model_path=args.llava_model_path,
        device=args.device,
        dtype=dtype,
        quantize=args.quantize,
        vision_encoder_path=args.vision_encoder_path,
        vision_encoder_model=args.vision_encoder_model,
        max_new_tokens=args.max_new_tokens,
    )

    all_metrics = {}

    # Evaluate on each provided dataset
    dataset_configs = [
        (args.coco_mcq,       "coco-mcq"),
        (args.voc2007_mcq,    "voc2007-mcq"),
        (args.synthetic_mcq,  "synthetic-mcq"),
    ]
    for csv_path, name in dataset_configs:
        if csv_path is None:
            continue
        logging.info(f"Evaluating {name} from {csv_path} ...")
        metrics = evaluate_llava_on_mcq(
            evaluator=evaluator,
            csv_path=csv_path,
            shuffle_options=args.shuffle_mcq_options,
            seed=args.seed,
            dataset_name=name,
        )

        # Save per-sample CSV
        pred_dir = os.path.join(log_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        sample_key = f"{name}-sample_results"
        if sample_key in metrics:
            df = pd.DataFrame(metrics.pop(sample_key))
            csv_out = os.path.join(pred_dir, f"{name}_predictions.csv")
            df.to_csv(csv_out, index=False)
            logging.info(f"Saved predictions to {csv_out}")

        all_metrics.update(metrics)

    # Print summary
    logging.info("=" * 60)
    logging.info("Evaluation complete. Summary:")
    for k, v in all_metrics.items():
        if isinstance(v, float):
            logging.info(f"  {k}: {v:.4f}")
        elif isinstance(v, dict):
            logging.info(f"  {k}: {v}")
    logging.info("=" * 60)

    # Save JSON
    results_path = os.path.join(log_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logging.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
