"""Diffusion operations used by the omics reconstruction loss."""

import math

import torch


def get_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -torch.arange(
            half,
            dtype=torch.float32,
            device=timesteps.device,
        )
        * (math.log(10000.0) / half)
    )
    phases = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat([torch.sin(phases), torch.cos(phases)], dim=1)


def cosine_beta_schedule(num_steps: int) -> torch.Tensor:
    s = 0.008
    timesteps = torch.arange(num_steps + 1, dtype=torch.float32)
    cumulative = torch.cos(
        ((timesteps / num_steps) + s) / (1 + s) * math.pi / 2
    ).square()
    alphas_cumprod = cumulative / cumulative[0]
    return 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
