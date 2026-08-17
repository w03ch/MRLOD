"""Datasets subtype counts and assignment output."""

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DATASET_CLUSTERS = {
    "tcga_blca_all_clean": 5,
    "tcga_brca_all_clean": 5,
    "tcga_coadread_all_clean": 8,
    "tcga_hnsc_all_clean": 2,
    "tcga_kirc_all_clean": 4,
    "tcga_kirp_all_clean": 4,
    "tcga_lgg_all_clean": 3,
    "tcga_luad_all_clean": 3,
    "tcga_stad_all_clean": 5,
    "tcga_ucec_all_clean": 4,
}


def save_subtype_assignments(
    output_path: Path,
    slide_ids: Sequence[str],
    case_ids: Sequence[str],
    labels: np.ndarray,
) -> None:
    assignments = pd.DataFrame(
        {
            "slide_id": [str(value) for value in slide_ids],
            "case_id": [str(value) for value in case_ids],
            "mrlod_subtype": np.asarray(labels, dtype=np.int64),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output_path, index=False, lineterminator="\n")
