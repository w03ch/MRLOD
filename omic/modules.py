"""Neural-network building blocks for molecular-only MRLOD."""

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn


class OmicsPerModEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class OmicsSharedProj(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)


class LowRankResidualFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        rank: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        rank = min(rank, dim)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, rank),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rank, dim),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + torch.sigmoid(self.scale) * self.net(features)


class OmicsLiteFusion(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        dropout: float = 0.1,
        rank: int = 64,
    ) -> None:
        super().__init__()
        rank = min(rank, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.score = nn.Sequential(
            nn.Linear(latent_dim, rank),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rank, 1),
        )
        self.ffn = LowRankResidualFFN(
            dim=latent_dim,
            rank=rank,
            dropout=dropout,
        )
        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        normalized = self.norm(tokens)
        weights = torch.softmax(
            self.score(normalized).squeeze(-1),
            dim=-1,
        )
        mixture = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        consensus = self.out_norm(self.ffn(mixture))
        return consensus.unsqueeze(1), {"omics_weights": weights}


class ZTModulatedMLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int,
        time_dim: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.modulation_proj = nn.Linear(
            latent_dim + time_dim,
            input_dim * 2 + output_dim,
        )
        self.modulation_proj.small_init = True
        self.modulation_proj.gate_bias_start = input_dim * 2
        self.update = nn.Sequential(
            nn.SiLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        global_representation: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        condition = torch.cat(
            [global_representation, time_embedding],
            dim=-1,
        )
        shift, scale, gate = self.modulation_proj(condition).split(
            [self.input_dim, self.input_dim, self.output_dim],
            dim=-1,
        )
        hidden = self.norm(features)
        hidden = hidden * (1.0 + scale) + shift
        return gate * self.update(hidden)


class ConditionalDecoderMLP(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        latent_dim: int,
        time_dim: int,
        hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        trunk_dim = hidden_dims[0]
        self.input_proj = nn.Linear(feature_dim, trunk_dim)
        self.z_proj = nn.Linear(latent_dim, trunk_dim)
        self.hidden_blocks = nn.ModuleList()
        input_dim = trunk_dim
        for output_dim in hidden_dims[1:]:
            self.hidden_blocks.append(
                ZTModulatedMLPBlock(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    latent_dim=latent_dim,
                    time_dim=time_dim,
                )
            )
            input_dim = output_dim
        self.output_proj = nn.Linear(hidden_dims[-1], feature_dim)

    def forward(
        self,
        corrupted_features: torch.Tensor,
        global_representation: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        hidden = (
            self.input_proj(corrupted_features)
            + self.z_proj(global_representation)
        )
        for block in self.hidden_blocks:
            hidden = block(
                hidden,
                global_representation,
                time_embedding,
            )
        return self.output_proj(hidden)
