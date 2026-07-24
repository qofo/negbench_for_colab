import logging
import re
from typing import List

logger = logging.getLogger(__name__)

OPTION_LABELS = ["A", "B", "C", "D"]

_PRIORITY_PATTERNS: List[str] = [
    r"(?:correct\s+)?answer\s+(?:is|:)\s*[:\(]?\s*([A-D])\b",
    r"(?:choose|select|pick|go\s+with)\s+(?:option\s+)?[:\(]?\s*([A-D])\b",
    r"(?:option|choice)\s+([A-D])\b",
    r"^\s*\(\s*([A-D])\s*\)",
    r"^\s*([A-D])\s*[\.\:\)]\s",
    r"^\s*([A-D])\s*$",
]

_BOUNDARY_RE = re.compile(
    r"(?:^|[\s\(\[,;!?])([A-D])(?=$|[\s\.\,\:\;\)\]\}!?])",
    re.MULTILINE,
)

def parse_option_robust(text: str, option_labels: List[str] = None) -> int:
    """
    Extract the predicted option index from generated text.
    """
    if option_labels is None:
        option_labels = OPTION_LABELS
    label_set = {lbl.upper() for lbl in option_labels}
    text_upper = text.strip().upper()

    for pattern in _PRIORITY_PATTERNS:
        m = re.search(pattern, text_upper, re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            if letter in label_set:
                return option_labels.index(letter)

    matches = list(_BOUNDARY_RE.finditer(text_upper))
    valid = [m for m in matches if m.group(1) in label_set]
    if valid:
        return option_labels.index(valid[-1].group(1))

    logger.warning("Could not parse option from: %r  -> defaulting to index 0.", text)
    return 0
