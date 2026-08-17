"""Assign MRLOD subtypes from saved global latent representations."""

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from clustering import DATASET_CLUSTERS, save_subtype_assignments


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PCA_COMPONENTS = 20
KMEANS_N_INIT = 50
SEED = 0


def _load_latents(
    latent_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not latent_path.is_file():
        raise FileNotFoundError(f"Latent archive not found: {latent_path}")
    with np.load(latent_path) as archive:
        required = {"global_latent_features", "slide_ids", "case_ids"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"{latent_path} is missing arrays: {sorted(missing)}"
            )
        latents = np.asarray(
            archive["global_latent_features"],
            dtype=np.float32,
        )
        slide_ids = archive["slide_ids"].astype(str)
        case_ids = archive["case_ids"].astype(str)
    if latents.ndim != 2:
        raise ValueError(
            f"Latents must be a two-dimensional array: {latent_path}"
        )
    if not np.isfinite(latents).all():
        raise ValueError(f"Latents contain non-finite values: {latent_path}")
    if len(latents) != len(slide_ids) or len(latents) != len(case_ids):
        raise ValueError(
            f"Latents and identifiers have inconsistent lengths: {latent_path}"
        )
    if len(set(slide_ids.tolist())) != len(slide_ids):
        raise ValueError(f"Duplicate slide_ids in: {latent_path}")
    if len(set(case_ids.tolist())) != len(case_ids):
        raise ValueError(f"Duplicate case_ids in: {latent_path}")
    if min(latents.shape) < PCA_COMPONENTS:
        raise ValueError(
            f"PCA-{PCA_COMPONENTS} requires at least {PCA_COMPONENTS} "
            f"samples and features: {latent_path}"
        )
    return latents, slide_ids, case_ids


def assign_subtypes(
    source_root: Path,
    output_root: Path,
    datasets: Optional[Sequence[str]] = None,
) -> None:
    """PCA and k-means."""
    targets = (
        list(DATASET_CLUSTERS)
        if datasets is None
        else list(dict.fromkeys(datasets))
    )
    for dataset_name in targets:
        latent_path = source_root / dataset_name / "mrlod_latent.npz"
        latents, slide_ids, case_ids = _load_latents(latent_path)
        reduced = PCA(
            n_components=PCA_COMPONENTS,
            svd_solver="full",
            random_state=SEED,
        ).fit_transform(latents)
        labels = KMeans(
            n_clusters=DATASET_CLUSTERS[dataset_name],
            n_init=KMEANS_N_INIT,
            random_state=SEED,
        ).fit_predict(reduced).astype(np.int64)
        output_path = (
            output_root / dataset_name / "mrlod_subtypes.csv"
        )
        save_subtype_assignments(
            output_path,
            slide_ids.tolist(),
            case_ids.tolist(),
            labels,
        )
        print(f"Saved assignments: {output_path}")


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Directory containing per-cohort mrlod_latent.npz files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory for per-cohort mrlod_subtypes.csv files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_CLUSTERS),
        default=None,
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    source_root = (
        args.source_root
        if args.source_root is not None
        else REPOSITORY_ROOT / "artifacts" / "omic"
    )
    output_root = (
        args.output_root
        if args.output_root is not None
        else REPOSITORY_ROOT / "outputs" / "omic"
    )
    assign_subtypes(
        source_root.resolve(),
        output_root.resolve(),
        args.datasets,
    )


if __name__ == "__main__":
    main()
