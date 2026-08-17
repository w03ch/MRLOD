"""Multimodal data loading and deterministic execution."""

import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MODALITIES = ("cnv", "rnaseq", "meth", "image", "text")
OMICS_SUFFIXES = {
    "cnv": "_cnv",
    "rnaseq": "_rnaseq",
    "meth": "_meth",
}
TEXT_EMBEDDING_COLUMNS = tuple(f"emb_{index}" for index in range(768))


def _cohort_paths(
    data_dir: str,
    dataset_names: Optional[Sequence[str]],
) -> Sequence[Path]:
    molecular_directory = Path(data_dir)
    if dataset_names is None:
        return sorted(molecular_directory.glob("*.csv"))
    return [
        molecular_directory / f"{name}.csv"
        for name in dict.fromkeys(dataset_names)
    ]


def _load_omics(
    cohort_path: Path,
) -> tuple[Dict[str, np.ndarray], Sequence[str], Sequence[str]]:
    cohort_frame = pd.read_csv(cohort_path, low_memory=False)
    slide_ids = cohort_frame["slide_id"].astype(str).str.strip().tolist()
    case_ids = cohort_frame["case_id"].astype(str).str.strip().tolist()
    omics_arrays = {
        modality: cohort_frame.loc[
            :,
            [
                column
                for column in cohort_frame.columns
                if str(column).endswith(suffix)
            ],
        ].to_numpy(dtype=np.float32)
        for modality, suffix in OMICS_SUFFIXES.items()
    }
    return omics_arrays, slide_ids, case_ids


def _load_image_features(path: str) -> Dict[str, np.ndarray]:
    with Path(path).open("rb") as handle:
        feature_data = pickle.load(handle)
    filenames = [str(value) for value in feature_data["filenames"]]
    embeddings = np.asarray(feature_data["embeddings"], dtype=np.float32)
    return {
        filename: embeddings[index]
        for index, filename in enumerate(filenames)
    }


def _load_text_features(path: str) -> Dict[str, np.ndarray]:
    text_frame = pd.read_csv(path, low_memory=False)
    case_ids = text_frame["case_id"].astype(str).str.strip().tolist()
    embeddings = text_frame.loc[
        :, list(TEXT_EMBEDDING_COLUMNS)
    ].to_numpy(dtype=np.float32)
    return {
        case_id: embeddings[index]
        for index, case_id in enumerate(case_ids)
    }


def _align_features(
    feature_map: Mapping[str, np.ndarray],
    sample_ids: Sequence[str],
) -> np.ndarray:
    return np.stack(
        [feature_map[str(sample_id)] for sample_id in sample_ids],
        axis=0,
    ).astype(np.float32, copy=False)


def set_seed(seed: int = 0) -> None:
    """Configure the deterministic CUDA execution."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def preprocess_multimodal_data(
    data_dir: str,
    image_feature_path: str,
    text_feature_path: str,
    dataset_names: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load complete, sample-aligned five-modality prepared cohorts."""
    cohort_paths = _cohort_paths(data_dir, dataset_names)
    image_features_by_id = _load_image_features(image_feature_path)
    text_features_by_id = _load_text_features(text_feature_path)
    processed_data: Dict[str, Dict[str, Any]] = {}

    for cohort_path in cohort_paths:
        print(f"Loading {cohort_path.name}...")
        omics_arrays, slide_ids, case_ids = _load_omics(cohort_path)
        cohort_data: Dict[str, Any] = dict(omics_arrays)
        cohort_data["image"] = _align_features(
            image_features_by_id,
            slide_ids,
        )
        cohort_data["text"] = _align_features(
            text_features_by_id,
            case_ids,
        )
        cohort_data["slide_ids"] = list(slide_ids)
        cohort_data["case_ids"] = list(case_ids)
        processed_data[cohort_path.stem] = cohort_data
        print(f"  Loaded {len(slide_ids)} aligned samples.")
    return processed_data


class MultimodalDataset(Dataset):
    """In-memory, sample-aligned dataset for the five MRLOD modalities."""

    def __init__(self, data_dict: Mapping[str, Any]) -> None:
        self.tensor_data = {
            modality: torch.as_tensor(
                data_dict[modality],
                dtype=torch.float32,
            )
            for modality in MODALITIES
        }
        self.slide_ids = [
            str(value) for value in data_dict["slide_ids"]
        ]
        self.case_ids = [
            str(value) for value in data_dict["case_ids"]
        ]
        expected = len(self.slide_ids)
        if len(self.case_ids) != expected:
            raise ValueError("slide_ids and case_ids must have equal length")
        for modality, values in self.tensor_data.items():
            if len(values) != expected:
                raise ValueError(
                    f"{modality} contains {len(values)} rows; expected {expected}"
                )

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            modality: values[index]
            for modality, values in self.tensor_data.items()
        }
