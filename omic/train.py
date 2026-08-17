"""Train the MRLOD-Omics experiment and save fused latents."""

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
if os.name == "nt":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import sklearn
import torch
from torch.utils.data import DataLoader

from clustering import DATASET_CLUSTERS
from losses import (
    cluster_assignment_contrastive_loss,
    modality_entropy_hinge_loss,
)
from model import MRLOD
from utils import MultimodalDataset, preprocess_omics_data, set_seed


MODULE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_ROOT.parent

LATENT_DIM = 256
DIFFUSION_STEPS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
FEATURE_MASK_PROBABILITY = 0.5
EPOCHS = 250
CLUSTER_PRETRAIN_EPOCHS = 50
ENTROPY_LOSS_WEIGHT = 1.0
CLUSTER_LOSS_WEIGHT = 0.25
SEED = 0

PCA_COMPONENTS = 20
KMEANS_N_INIT = 50


def require_cuda() -> None:
    """Fail before loading data when the required CUDA runtime is unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "MRLOD-Omics training requires a CUDA-capable GPU and "
            "CUDA-enabled PyTorch. CPU training is not supported by this "
            "release."
        )


def _make_loader(
    dataset: MultimodalDataset,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=True,
    )


def _batch_modalities(
    batch: Mapping[str, Any],
    model: MRLOD,
) -> Dict[str, torch.Tensor]:
    return {
        modality: batch[modality].to(
            "cuda",
            non_blocking=True,
        )
        for modality in model.modalities
    }


def train_model(
    dataset_name: str,
    cohort_data: Mapping[str, Any],
) -> Tuple[MRLOD, MultimodalDataset]:
    set_seed(SEED)
    dataset = MultimodalDataset(cohort_data)
    if len(dataset) < BATCH_SIZE:
        raise ValueError(
            f"{dataset_name} has {len(dataset)} samples, fewer than the "
            f"configured batch size {BATCH_SIZE}"
        )
    input_dims = {
        modality: int(cohort_data[modality].shape[1])
        for modality in MRLOD.MODALITIES
    }
    model = MRLOD(
        input_dims=input_dims,
        n_clusters=DATASET_CLUSTERS[dataset_name],
        latent_dim=LATENT_DIM,
        diffusion_steps=DIFFUSION_STEPS,
        feature_mask_probability=FEATURE_MASK_PROBABILITY,
    ).to("cuda")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )
    loader = _make_loader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    print(
        f"\nTraining molecular MRLOD on {dataset_name}: "
        f"{len(dataset)} samples, {EPOCHS} epochs"
    )
    for epoch_index in range(EPOCHS):
        model.train()
        total_sum = 0.0
        for batch in loader:
            batch_data = _batch_modalities(batch, model)
            optimizer.zero_grad(set_to_none=True)
            global_representation, modality_tokens = (
                model.get_global_and_token_embeddings(batch_data)
            )
            diffusion_loss = model.omics_reconstruction_loss(
                batch_data,
                global_representation,
            )
            entropy_loss = modality_entropy_hinge_loss(
                modality_tokens["modality_weights"]
            )
            cluster_loss = global_representation.new_zeros(())
            if epoch_index >= CLUSTER_PRETRAIN_EPOCHS:
                cluster_loss = cluster_assignment_contrastive_loss(
                    model.get_cluster_logits(
                        global_representation,
                        modality_tokens,
                    )
                )
            total_loss = (
                diffusion_loss
                + ENTROPY_LOSS_WEIGHT * entropy_loss
                + CLUSTER_LOSS_WEIGHT * cluster_loss
            )
            total_loss.backward()
            optimizer.step()
            total_sum += float(total_loss.detach().item())

        epoch_number = epoch_index + 1
        if epoch_number in {1, CLUSTER_PRETRAIN_EPOCHS, EPOCHS}:
            print(
                f"  Epoch {epoch_number:03d}/{EPOCHS:03d} | "
                f"total={total_sum / len(loader):.4f}"
            )
    return model, dataset


def extract_global_latents(
    model: MRLOD,
    dataset: MultimodalDataset,
) -> np.ndarray:
    loader = _make_loader(
        dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
    )
    model.eval()
    global_latents = []
    with torch.no_grad():
        for batch in loader:
            batch_data = _batch_modalities(batch, model)
            global_representation, _ = (
                model.get_global_and_token_embeddings(batch_data)
            )
            global_latents.append(global_representation.cpu())
    return torch.cat(global_latents, dim=0).numpy()


def save_latents(
    dataset_name: str,
    dataset: MultimodalDataset,
    global_latents: np.ndarray,
    output_root: Path,
) -> None:
    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mrlod_latent.npz"
    np.savez_compressed(
        output_path,
        global_latent_features=global_latents,
        slide_ids=np.asarray(dataset.slide_ids, dtype=str),
        case_ids=np.asarray(dataset.case_ids, dtype=str),
    )
    print(f"  Saved fused representations: {output_path}")


def save_model_weights(
    dataset_name: str,
    model: MRLOD,
    dataset: MultimodalDataset,
    cohort_data: Mapping[str, Any],
    output_root: Path,
) -> None:
    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mrlod_model.pt"
    checkpoint = {
        "format_version": 1,
        "dataset_name": dataset_name,
        "model_class": "MRLOD",
        "model_config": {
            "input_dims": {
                modality: int(cohort_data[modality].shape[1])
                for modality in MRLOD.MODALITIES
            },
            "n_clusters": DATASET_CLUSTERS[dataset_name],
            "latent_dim": LATENT_DIM,
            "diffusion_steps": DIFFUSION_STEPS,
            "feature_mask_probability": FEATURE_MASK_PROBABILITY,
        },
        "training_config": {
            "epochs": EPOCHS,
            "cluster_pretrain_epochs": CLUSTER_PRETRAIN_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "entropy_loss_weight": ENTROPY_LOSS_WEIGHT,
            "cluster_loss_weight": CLUSTER_LOSS_WEIGHT,
            "seed": SEED,
        },
        "slide_ids": list(dataset.slide_ids),
        "case_ids": list(dataset.case_ids),
        "model_state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
    }
    torch.save(checkpoint, output_path)
    print(f"  Saved model weights: {output_path}")


def save_run_config(
    dataset_name: str,
    dataset: MultimodalDataset,
    output_root: Path,
) -> None:
    """Record the configuration and runtime beside each run."""
    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "omic",
        "dataset": dataset_name,
        "n_samples": len(dataset),
        "n_clusters": DATASET_CLUSTERS[dataset_name],
        "training": {
            "latent_dim": LATENT_DIM,
            "diffusion_steps": DIFFUSION_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "feature_mask_probability": FEATURE_MASK_PROBABILITY,
            "epochs": EPOCHS,
            "cluster_pretrain_epochs": CLUSTER_PRETRAIN_EPOCHS,
            "entropy_loss_weight": ENTROPY_LOSS_WEIGHT,
            "cluster_loss_weight": CLUSTER_LOSS_WEIGHT,
            "seed": SEED,
        },
        "clustering": {
            "pca_components": PCA_COMPONENTS,
            "pca_svd_solver": "full",
            "kmeans_n_init": KMEANS_N_INIT,
            "random_state": SEED,
        },
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    output_path = output_dir / "run_config.json"
    output_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"  Saved run configuration: {output_path}")


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_CLUSTERS),
        default=None,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPOSITORY_ROOT / "data",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "omic",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    require_cuda()
    requested_datasets = (
        args.datasets
        if args.datasets is not None
        else list(DATASET_CLUSTERS)
    )
    data_root = args.data_root.resolve()
    cohort_database = preprocess_omics_data(
        str(data_root / "OMIC"),
        dataset_names=requested_datasets,
    )
    print("Using device: cuda")
    for dataset_name in requested_datasets:
        model, dataset = train_model(
            dataset_name,
            cohort_database[dataset_name],
        )
        save_latents(
            dataset_name,
            dataset,
            extract_global_latents(model, dataset),
            args.output_root.resolve(),
        )
        save_model_weights(
            dataset_name,
            model,
            dataset,
            cohort_database[dataset_name],
            args.output_root.resolve(),
        )
        save_run_config(
            dataset_name,
            dataset,
            args.output_root.resolve(),
        )


if __name__ == "__main__":
    main()
