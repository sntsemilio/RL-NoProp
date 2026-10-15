"""Prepared categorical interfaces (sections 6--8); no S/A training is launched."""

from dataclasses import dataclass

import torch
from torch import nn


def categorical_posterior(base_logits, z, codes, rho):
    if not 0 <= rho < 1:
        raise ValueError("rho outside channel domain")
    evidence = -((z[..., None] - rho**0.5 * codes) ** 2) / (2 * (1 - rho))
    return (base_logits + evidence).softmax(-1)


@torch.no_grad()
def project_target(reward, terminated, next_mass, atoms, gamma):
    """Genuine terminal flag only; external time truncations retain bootstrap."""
    if next_mass.requires_grad:
        raise ValueError("Target must be frozen before local updates")
    y = reward[:, None] + gamma * (1 - terminated[:, None]) * atoms
    clipped = y.clamp(atoms[0], atoms[-1])
    b = (clipped - atoms[0]) / (atoms[1] - atoms[0])
    lo = b.floor().long().clamp(0, len(atoms) - 1)
    hi = b.ceil().long().clamp(0, len(atoms) - 1)
    out = torch.zeros_like(next_mass)
    out.scatter_add_(
        1, lo, next_mass * torch.where(lo == hi, torch.ones_like(b), hi - b)
    )
    out.scatter_add_(
        1, hi, next_mass * torch.where(lo == hi, torch.zeros_like(b), b - lo)
    )
    return out


@torch.no_grad()
def corrupt_frozen_target(mass, codes, rho, category_rng, noise_rng):
    if mass.requires_grad:
        raise ValueError("Target must be detached")
    index = torch.multinomial(mass, 1, generator=category_rng).squeeze(-1)
    z = rho**0.5 * codes[index] + (1 - rho) ** 0.5 * torch.randn(
        index.shape, generator=noise_rng, device=mass.device
    )
    return index, z


@dataclass(frozen=True)
class RealTransition:
    observation: object
    action: int
    reward: float
    final_observation: object
    terminated: bool
    truncated: bool
    emulated_frames: int


def visual_encoder():
    """Own a new encoder per block; never share this module between optimizers."""
    return nn.Sequential(
        nn.Conv2d(4, 32, 8, 4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
    )


def launch_later(stage):
    raise RuntimeError(
        f"{stage} is pending. No sequential/Atari experiment is part of this launch; A3 needs a measured budget."
    )
