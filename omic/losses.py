"""Losses used by the fixed molecular-only MRLOD experiment."""

import math
from typing import Mapping

import torch
import torch.nn.functional as F


VIEW_NAMES = ("cnv", "rnaseq", "meth")
ENTROPY_FLOOR_RATIO = 0.5
EPS = 1e-8


def cluster_assignment_contrastive_loss(
    cluster_logits: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    global_assignment = torch.softmax(cluster_logits["global"], dim=-1)
    n_clusters = global_assignment.size(-1)
    targets = torch.arange(n_clusters, device=global_assignment.device)
    global_columns = F.normalize(
        global_assignment.transpose(0, 1),
        dim=-1,
        eps=EPS,
    )

    contrastive = global_assignment.new_zeros(())
    assignments = [global_assignment]
    for view_name in VIEW_NAMES:
        view_assignment = torch.softmax(
            cluster_logits[view_name],
            dim=-1,
        )
        assignments.append(view_assignment)
        view_columns = F.normalize(
            view_assignment.transpose(0, 1),
            dim=-1,
            eps=EPS,
        )
        similarities = view_columns @ global_columns.transpose(0, 1)
        contrastive = contrastive + 0.5 * (
            F.cross_entropy(similarities, targets)
            + F.cross_entropy(similarities.transpose(0, 1), targets)
        )
    contrastive = contrastive / len(VIEW_NAMES)

    marginal = torch.stack(assignments, dim=0).mean(dim=(0, 1))
    marginal_entropy = -(
        marginal * marginal.clamp_min(EPS).log()
    ).sum()
    normalized_entropy = marginal_entropy / math.log(float(n_clusters))
    collapse_penalty = F.relu(
        ENTROPY_FLOOR_RATIO - normalized_entropy
    ).square()
    return contrastive + collapse_penalty


def modality_entropy_hinge_loss(
    modality_weights: torch.Tensor,
) -> torch.Tensor:
    n_modalities = modality_weights.size(-1)
    entropy = -(
        modality_weights
        * modality_weights.clamp_min(1e-12).log()
    ).sum(dim=-1)
    entropy_floor = ENTROPY_FLOOR_RATIO * math.log(float(n_modalities))
    return F.relu(entropy_floor - entropy).square().mean()
