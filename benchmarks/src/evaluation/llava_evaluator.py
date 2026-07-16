"""
LLaVA Modular Evaluator
=======================
Loads a locally stored LLaVA checkpoint and exposes each sub-module
separately so that the vision encoder can be replaced with:
  - the original SigLIP/CLIP that shipped with LLaVA
  - a fine-tuned CLIP checkpoint (e.g., NegCLIP, CC12M-NegFull)
  - any OpenCLIP-compatible model

Supported LLaVA variants
-------------------------
- LLaVA-1.5  (llava-hf/llava-1.5-*)
- LLaVA-NeXT (llava-hf/llava-v1.6-*)
- Any model that follows the `LlavaForConditionalGeneration` HuggingFace API.

Usage
-----
See `eval_negation_llava.py` for the full evaluation pipeline, or use
this class directly:

    evaluator = LLaVAModularEvaluator(
        model_path="/path/to/llava-1.5-7b",
        device="cuda",
        # Optional: swap the vision tower
        vision_encoder_path=None,            # keep original
        vision_encoder_model="ViT-L-14",     # used only when swapping
    )
    answer = evaluator.generate_mcq_answer(image_pil, captions)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: load an OpenCLIP-compatible vision encoder from a checkpoint
# ---------------------------------------------------------------------------

def _load_openclip_vision_encoder(
    model_name: str,
    pretrained: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, callable]:
    """
    Load an OpenCLIP vision tower (image encoder + preprocess) that can be
    dropped into the LLaVA model in place of its original vision tower.

    Args:
        model_name:  OpenCLIP architecture string, e.g. "ViT-B-32".
        pretrained:  Tag (e.g. "openai") or absolute path to a .pt/.pth file.
        device:      Target device.

    Returns:
        (visual_encoder, preprocess_val)
    """
    from open_clip import create_model_and_transforms
    model, _, preprocess_val = create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
        output_dict=True,
    )
    model.eval()
    return model.visual, preprocess_val


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LLaVAModularEvaluator:
    """
    Modular LLaVA evaluator with optional vision-encoder hot-swap.

    Architecture breakdown
    ----------------------
    self.vision_tower    : nn.Module  -- encodes images -> patch feature grid
    self.mm_projector    : nn.Module  -- projects vision features -> LLM input space
    self.language_model  : nn.Module  -- causal LLM (Vicuna, Mistral, ...)
    self.processor       : LlavaProcessor -- tokeniser + image preprocessor
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        # --- optional vision-encoder swap ---
        vision_encoder_path: Optional[str] = None,
        vision_encoder_model: Optional[str] = None,   # e.g. "ViT-B-32"
        # --- generation ---
        max_new_tokens: int = 16,
        temperature: float = 0.0,
    ):
        """
        Args:
            model_path:
                Path to the local LLaVA checkpoint directory
                (must contain config.json, tokenizer files, etc.).
            device:
                PyTorch device string.
            dtype:
                Model weight dtype. float16 recommended for GPU inference.
            vision_encoder_path:
                If not None, path to an OpenCLIP-compatible checkpoint (.pt/.pth)
                whose *visual* sub-module will replace LLaVA's original vision tower.
                If None, the original vision tower is kept.
            vision_encoder_model:
                OpenCLIP architecture string (e.g. "ViT-B-32") required when
                vision_encoder_path is provided.
            max_new_tokens:
                Maximum tokens to generate per MCQ answer.
            temperature:
                Sampling temperature; 0.0 means greedy decoding.
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # ------------------------------------------------------------------
        # 1. Load the full LLaVA model via HuggingFace transformers
        # ------------------------------------------------------------------
        logger.info(f"Loading LLaVA from {model_path} ...")
        try:
            from transformers import LlavaForConditionalGeneration, LlavaProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers >= 4.36 is required for LLaVA support. "
                "Install with: pip install transformers>=4.36"
            ) from exc

        self.processor = LlavaProcessor.from_pretrained(model_path)
        self._full_model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self._full_model.eval()

        # ------------------------------------------------------------------
        # 2. Expose sub-modules as named attributes
        # ------------------------------------------------------------------
        self.vision_tower: torch.nn.Module = self._full_model.vision_tower
        self.mm_projector: torch.nn.Module = self._full_model.multi_modal_projector
        self.language_model: torch.nn.Module = self._full_model.language_model

        logger.info(
            f"  vision_tower  : {type(self.vision_tower).__name__}\n"
            f"  mm_projector  : {type(self.mm_projector).__name__}\n"
            f"  language_model: {type(self.language_model).__name__}"
        )

        # ------------------------------------------------------------------
        # 3. Optionally swap the vision tower
        # ------------------------------------------------------------------
        self._external_preprocess = None  # set when swapping encoder

        if vision_encoder_path is not None:
            if vision_encoder_model is None:
                raise ValueError(
                    "vision_encoder_model (e.g. 'ViT-B-32') must be specified "
                    "when vision_encoder_path is provided."
                )
            self._swap_vision_encoder(vision_encoder_model, vision_encoder_path)

    # ------------------------------------------------------------------
    # Vision encoder hot-swap
    # ------------------------------------------------------------------

    def _swap_vision_encoder(self, model_name: str, pretrained: str) -> None:
        """
        Replace the LLaVA vision tower with an OpenCLIP visual encoder.

        The replacement encoder must produce features of the same hidden
        dimension as the original vision tower, otherwise the mm_projector
        will produce incorrect outputs. A shape check is logged.

        Args:
            model_name: OpenCLIP architecture (e.g. "ViT-B-32").
            pretrained: Tag or path to the checkpoint.
        """
        logger.info(
            f"Swapping vision tower with OpenCLIP {model_name} "
            f"from '{pretrained}' ..."
        )
        new_visual, preprocess = _load_openclip_vision_encoder(
            model_name, pretrained, self.device
        )
        new_visual = new_visual.to(self.device, dtype=self.dtype)

        # Log dimension compatibility info
        orig_hidden = getattr(
            self._full_model.config.vision_config, "hidden_size", "unknown"
        )
        new_hidden = getattr(new_visual, "output_dim", "unknown")
        if orig_hidden != "unknown" and new_hidden != "unknown" and orig_hidden != new_hidden:
            logger.warning(
                f"Vision tower output dim mismatch: "
                f"original={orig_hidden}, new={new_hidden}. "
                "The mm_projector may need to be re-trained."
            )
        else:
            logger.info(f"  vision hidden dim: {new_hidden} (original: {orig_hidden})")

        # Patch the full model in-place so generate() still works end-to-end
        self._full_model.vision_tower = new_visual
        self.vision_tower = new_visual
        self._external_preprocess = preprocess
        logger.info("Vision tower swap complete.")

    # ------------------------------------------------------------------
    # MCQ answer generation
    # ------------------------------------------------------------------

    def generate_mcq_answer(
        self,
        image: Image.Image,
        captions: List[str],
        option_labels: Optional[List[str]] = None,
    ) -> Tuple[int, str]:
        """
        Present the image + MCQ prompt to the LLaVA model and return the
        predicted answer index.

        Args:
            image:
                A PIL Image.
            captions:
                List of N caption strings (answer choices).
            option_labels:
                Labels for each option, e.g. ["A", "B", "C", "D"].
                Defaults to ["A", "B", "C", "D"] for N=4.

        Returns:
            (predicted_index, raw_generated_text)
            predicted_index is the 0-based index into captions.
        """
        n = len(captions)
        if option_labels is None:
            option_labels = [chr(ord("A") + i) for i in range(n)]

        # Build the MCQ prompt
        options_str = "\n".join(
            f"({lbl}) {cap}" for lbl, cap in zip(option_labels, captions)
        )
        prompt_text = (
            "Which caption best describes the image?\n"
            f"{options_str}\n"
            "Answer with only the letter of the correct option."
        )

        # Use the LLaVA conversation template
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        # Tokenize
        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device)

        # Cast pixel values to model dtype
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        # Greedy / temperature-based decode
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
        )
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self._full_model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        raw_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        # Parse the predicted option letter
        predicted_index = self._parse_option(raw_text, option_labels)
        return predicted_index, raw_text

    @staticmethod
    def _parse_option(text: str, option_labels: List[str]) -> int:
        """
        Extract the predicted option index from the generated text.
        Tries to find the first occurrence of any label letter.
        Falls back to index 0 on parse failure.
        """
        text_upper = text.upper().strip()
        for idx, label in enumerate(option_labels):
            if label in text_upper:
                return idx
        logger.warning(
            f"Could not parse option from generated text: '{text}'. "
            "Falling back to index 0."
        )
        return 0

    # ------------------------------------------------------------------
    # Convenience: encode image only (for embedding-style analysis)
    # ------------------------------------------------------------------

    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run only the vision tower and return normalised image features.
        Useful for comparing vision representations across encoder swap experiments.

        Args:
            image_tensor: (B, C, H, W) preprocessed image batch.

        Returns:
            (B, D) L2-normalised feature vectors.
        """
        with torch.no_grad():
            feats = self.vision_tower(image_tensor)
            # Different vision towers expose features differently
            if hasattr(feats, "last_hidden_state"):
                feats = feats.last_hidden_state[:, 0]   # CLS token
            elif hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            else:
                feats = feats                             # assume (B, D) already
        return F.normalize(feats.float(), dim=-1)
