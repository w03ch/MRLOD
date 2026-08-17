"""MRLOD model."""

from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion import cosine_beta_schedule, get_timestep_embedding
from modules import (
    ConditionalDecoderMLP,
    OmicsLiteFusion,
    OmicsPerModEncoder,
    OmicsSharedProj,
)


class MRLOD(nn.Module):
    OMICS_MODALITIES = ("cnv", "rnaseq", "meth")
    MODALITIES = OMICS_MODALITIES
    CLUSTER_VIEWS = OMICS_MODALITIES
    TIMESTEP_EMBEDDING_DIM = 128

    def __init__(
        self,
        input_dims: Mapping[str, int],
        n_clusters: int,
        latent_dim: int = 256,
        diffusion_steps: int = 1000,
        feature_mask_probability: float = 0.5,
    ) -> None:
        super().__init__()
        self.modalities = list(self.MODALITIES)
        self.cluster_views = list(self.CLUSTER_VIEWS)
        self.T = diffusion_steps
        self.x_mask_prob = feature_mask_probability
        self.cluster_head = nn.Linear(
            latent_dim,
            n_clusters,
            bias=False,
        )

        self.omics_per_mod_encoders = nn.ModuleDict()
        for modality in ("rnaseq", "cnv", "meth"):
            self.omics_per_mod_encoders[modality] = OmicsPerModEncoder(
                input_dim=input_dims[modality],
                hidden_dim=2048,
                output_dim=1024,
                dropout=0.1,
            )
        self.omics_shared_proj = OmicsSharedProj(
            input_dim=1024,
            output_dim=latent_dim,
        )
        self.omics_fusion = OmicsLiteFusion(
            latent_dim=latent_dim,
            dropout=0.1,
            rank=64,
        )

        self.diffusion_decoders = nn.ModuleDict()
        for modality in self.OMICS_MODALITIES:
            self.diffusion_decoders[modality] = ConditionalDecoderMLP(
                feature_dim=input_dims[modality],
                latent_dim=latent_dim,
                time_dim=self.TIMESTEP_EMBEDDING_DIM,
                hidden_dims=(512, 512),
            )

        betas = cosine_beta_schedule(diffusion_steps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(alphas_cumprod),
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self._timestep_rng = torch.Generator(device="cpu")
        self._timestep_rng.manual_seed(0)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if getattr(module, "small_init", False):
                nn.init.normal_(module.weight, mean=0.0, std=1e-3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    gate_bias_start = getattr(
                        module,
                        "gate_bias_start",
                    )
                    with torch.no_grad():
                        module.bias[gate_bias_start:].fill_(0.2)
                return
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_global_and_token_embeddings(
        self,
        batch_data: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raw_tokens: Dict[str, torch.Tensor] = {}
        for modality in self.OMICS_MODALITIES:
            features = self.omics_per_mod_encoders[modality](
                batch_data[modality]
            )
            raw_tokens[modality] = self.omics_shared_proj(
                features
            ).unsqueeze(1)

        omics_sequence = torch.cat(
            [raw_tokens[name] for name in self.OMICS_MODALITIES],
            dim=1,
        )
        omics_consensus, omics_statistics = self.omics_fusion(
            omics_sequence
        )
        global_representation = omics_consensus.squeeze(1)
        omics_weights = omics_statistics["omics_weights"]
        modality_tokens = {
            name: raw_tokens[name].squeeze(1)
            for name in self.OMICS_MODALITIES
        }
        modality_tokens.update(
            {
                "omics": global_representation,
                "omics_weights": omics_weights,
                "modality_weights": omics_weights,
                "modality_entropy": -(
                    omics_weights
                    * omics_weights.clamp_min(1e-12).log()
                ).sum(dim=-1),
            }
        )
        return global_representation, modality_tokens

    def get_cluster_logits(
        self,
        global_representation: torch.Tensor,
        modality_tokens: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        logits = {
            "global": self.cluster_head(global_representation),
        }
        logits.update(
            {
                name: self.cluster_head(modality_tokens[name])
                for name in self.cluster_views
            }
        )
        return logits

    def _sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        indices = torch.arange(
            self.T,
            dtype=torch.float32,
            device=device,
        )
        center = (self.T - 1) / 2.0
        standard_deviation = max(self.T / 4.0, 1.0)
        weights = torch.exp(
            -0.5
            * ((indices - center) / standard_deviation).square()
        )
        sampled = torch.multinomial(
            (weights / weights.sum()).cpu(),
            num_samples=batch_size,
            replacement=True,
            generator=self._timestep_rng,
        )
        return sampled.to(device=device, dtype=torch.long)

    def omics_reconstruction_loss(
        self,
        batch_data: Mapping[str, torch.Tensor],
        global_representation: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = global_representation.size(0)
        device = global_representation.device
        timesteps = self._sample_timesteps(batch_size, device)
        time_embedding = get_timestep_embedding(
            timesteps,
            self.TIMESTEP_EMBEDDING_DIM,
        )
        total_loss = global_representation.new_zeros(())

        for modality in self.OMICS_MODALITIES:
            clean_features = batch_data[modality]
            noise = torch.randn_like(clean_features)
            sqrt_alpha_bar = self.sqrt_alphas_cumprod[timesteps].view(
                clean_features.shape[0],
                1,
            )
            sqrt_one_minus_alpha_bar = (
                self.sqrt_one_minus_alphas_cumprod[timesteps].view(
                    clean_features.shape[0],
                    1,
                )
            )
            corrupted_features = (
                sqrt_alpha_bar * clean_features
                + sqrt_one_minus_alpha_bar * noise
            )
            observed_mask = (
                torch.rand_like(corrupted_features) > self.x_mask_prob
            ).to(corrupted_features.dtype)
            predicted_clean = self.diffusion_decoders[modality](
                corrupted_features * observed_mask,
                global_representation,
                time_embedding,
            )
            total_loss = total_loss + F.mse_loss(
                predicted_clean,
                clean_features,
            )
        return total_loss
