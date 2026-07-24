import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

ORIGINAL_CAPTION_TYPES = ["gt", "hybrid", "positive", "negative"]

def build_dataset_index(dataset_csv: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv, sep=",")
    path_col_candidates = ["image_path", "filepath", "path", "file_path"]
    path_col = next((c for c in path_col_candidates if c in df.columns), None)
    if path_col is None:
        raise ValueError(f"Cannot locate image-path column in {dataset_csv}.")
    if path_col != "image_path":
        df = df.rename(columns={path_col: "image_path"})

    cap_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"caption_\d+", c)],
        key=lambda c: int(c.split("_")[1]),
    )
    if not cap_cols:
        raise ValueError(f"No caption_N columns found in {dataset_csv}.")

    df["_caption_set"] = df[cap_cols].apply(
        lambda row: frozenset(str(v) for v in row), axis=1
    )
    df["_cap_cols"] = [cap_cols] * len(df)
    return df

def _match_original_row(shuffled_captions: List[str], image_path: str, ds_df: pd.DataFrame) -> Optional[pd.Series]:
    target_set = frozenset(str(c) for c in shuffled_captions)
    candidates = ds_df[ds_df["image_path"] == image_path]
    
    if candidates.empty:
        basename = Path(image_path).name
        candidates = ds_df[ds_df["image_path"].apply(lambda p: Path(p).name) == basename]

    if candidates.empty:
        return None

    for _, row in candidates.iterrows():
        if row["_caption_set"] == target_set:
            return row
    return None

def build_perm(shuffled: List[str], original: List[str]) -> List[int]:
    perm = []
    remaining = list(enumerate(shuffled))
    for orig_cap in original:
        for j, (s_idx, s_cap) in enumerate(remaining):
            if orig_cap == s_cap:
                perm.append(s_idx)
                remaining.pop(j)
                break
        else:
            raise ValueError(f"Caption not found in shuffled list: {orig_cap!r}")
    return perm

def restore_original_order(
    shuffled_captions: List[str],
    shuffled_correct_answer: int,
    shuffled_predicted_answer: int,
    image_path: str,
    ds_df: pd.DataFrame,
) -> Tuple[List[str], int, int, List[str], bool]:
    orig_row = _match_original_row(shuffled_captions, image_path, ds_df)
    if orig_row is None:
        logger.warning(
            "No matching original row found for image '%s'. Keeping shuffled order.", image_path
        )
        return (
            shuffled_captions,
            shuffled_correct_answer,
            shuffled_predicted_answer,
            ["unknown"] * len(shuffled_captions),
            False,
        )

    cap_cols = orig_row["_cap_cols"]
    original_captions = [str(orig_row[c]) for c in cap_cols]
    orig_types = list(ORIGINAL_CAPTION_TYPES[: len(original_captions)])
    perm = build_perm(shuffled_captions, original_captions)

    def to_orig(shuffled_idx: int) -> int:
        return next(i for i, s in enumerate(perm) if s == shuffled_idx)

    orig_correct = to_orig(shuffled_correct_answer)
    orig_predicted = to_orig(shuffled_predicted_answer)

    return original_captions, orig_correct, orig_predicted, orig_types, True
