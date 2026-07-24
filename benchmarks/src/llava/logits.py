import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

from src.llava.parser import OPTION_LABELS

def get_option_token_ids(tokenizer, option_labels: List[str] = None) -> Dict[str, int]:
    if option_labels is None:
        option_labels = OPTION_LABELS
    token_ids: Dict[str, int] = {}
    for label in option_labels:
        resolved = False
        for candidate in [f" {label}", label, f"({label})"]:
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) == 1:
                token_ids[label] = ids[0]
                resolved = True
                break
        if not resolved:
            ids = tokenizer.encode(label, add_special_tokens=False)
            token_ids[label] = ids[0]
            logger.warning(
                "Option '%s' encodes to multiple tokens; using first token id=%d.",
                label, ids[0],
            )
    logger.info("Option token ids: %s", token_ids)
    return token_ids

def extract_option_logits(
    model,
    processor,
    image_path: str,
    captions: List[str],
    option_labels: List[str],
    option_token_ids: Dict[str, int],
    device: torch.device,
    dtype: torch.dtype,
    use_device_map: bool,
) -> np.ndarray:
    n = len(captions)
    labels = option_labels[:n]

    options_str = "\n".join(f"({lbl}) {cap}" for lbl, cap in zip(labels, captions))
    prompt_text = (
        "Which caption best describes the image?\n"
        f"{options_str}\n"
        "Answer with only the letter of the correct option."
    )
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    if hasattr(processor, "apply_chat_template"):
        text_prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
    else:
        text_prompt = processor.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

    pil_image = Image.open(image_path).convert("RGB")
    inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt")

    if not use_device_map:
        inputs = inputs.to(device)
    else:
        inputs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    first_logits = outputs.scores[0][0]
    raw = torch.tensor(
        [first_logits[option_token_ids[lbl]].item() for lbl in labels],
        dtype=torch.float32,
    )
    probs = F.softmax(raw, dim=0).cpu().numpy()
    return probs
