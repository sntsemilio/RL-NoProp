"""SGD local, con planes aleatorios independientes por réplica."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import LossName, stable_sigmoid


@dataclass
class FitResult:
    theta_1: np.ndarray | None
    theta_2: np.ndarray | None
    loss_history_1: np.ndarray | None
    loss_history_2: np.ndarray | None
    update_count_1: int
    update_count_2: int


def _fit_block_batched(
    actions: np.ndarray,
    rewards: np.ndarray,
    block: int,
    rho: float,
    loss: LossName,
    learning_rate: float,
    batch_indices: np.ndarray,
    noises: np.ndarray,
    initial_parameters: np.ndarray,
    record_every: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_replicas, num_steps, _batch_size = batch_indices.shape
    theta = np.asarray(initial_parameters, dtype=np.float64).copy()
    if theta.shape != (num_replicas, 3):
        raise ValueError("initial_parameters debe tener forma (réplicas, 3)")
    history = []
    rows = np.arange(num_replicas)[:, None]
    for step in range(num_steps):
        indices = batch_indices[:, step, :]
        action_batch = actions[rows, indices]
        reward_batch = rewards[rows, indices]
        if block == 1:
            z = noises[:, step, :]
        elif block == 2:
            z = np.sqrt(rho) * reward_batch + np.sqrt(1.0 - rho) * noises[:, step, :]
        else:
            raise ValueError("block debe ser 1 o 2")
        logits = theta[:, 0, None] * z + np.take_along_axis(
            theta[:, 1:], action_batch, axis=1
        )
        predictions = stable_sigmoid(logits)
        residual = predictions - reward_batch
        if loss == "mse":
            factors = residual * predictions * (1.0 - predictions)
            values = 0.5 * np.mean(residual**2, axis=1)
        elif loss == "bce":
            factors = residual
            values = np.mean(np.logaddexp(0.0, logits) - reward_batch * logits, axis=1)
        else:
            raise ValueError(f"Pérdida desconocida: {loss}")
        gradient = np.empty_like(theta)
        gradient[:, 0] = np.mean(factors * z, axis=1)
        gradient[:, 1] = np.mean(factors * (action_batch == 0), axis=1)
        gradient[:, 2] = np.mean(factors * (action_batch == 1), axis=1)
        theta -= learning_rate * gradient
        if step == 0 or (step + 1) % record_every == 0 or step + 1 == num_steps:
            history.append(np.column_stack((np.full(num_replicas, step + 1), values)))
    return theta, np.stack(history, axis=1)


def make_random_plan(
    seeds: list[int],
    num_steps: int,
    batch_size: int,
    num_observations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cada réplica posee su propio Generator; el tensor solo agrupa el cómputo."""

    indices = []
    noises = []
    for seed in seeds:
        sequence = np.random.SeedSequence(seed)
        batch_sequence, noise_sequence = sequence.spawn(2)
        batch_rng = np.random.default_rng(batch_sequence)
        noise_rng = np.random.default_rng(noise_sequence)
        indices.append(batch_rng.integers(0, num_observations, size=(num_steps, batch_size)))
        noises.append(noise_rng.standard_normal((num_steps, batch_size)))
    return np.stack(indices), np.stack(noises)


def fit_blocks_batched(
    actions: np.ndarray,
    rewards: np.ndarray,
    rho: float,
    loss: LossName,
    learning_rate: float,
    num_steps: int,
    batch_size: int,
    seeds_batch: list[int],
    seeds_noise_1: list[int],
    seeds_noise_2: list[int],
    train_block_1: bool = True,
    train_block_2: bool = True,
    initial_std: float = 0.0,
    seeds_initialization: list[int] | None = None,
    record_every: int = 100,
) -> FitResult:
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float64)
    if actions.shape != rewards.shape or actions.ndim != 2:
        raise ValueError("actions y rewards deben tener forma (réplicas, observaciones)")
    num_replicas, num_observations = actions.shape
    if not (len(seeds_batch) == len(seeds_noise_1) == len(seeds_noise_2) == num_replicas):
        raise ValueError("Cada réplica necesita semillas propias")
    batch_plans = []
    for seed in seeds_batch:
        batch_plans.append(
            np.random.default_rng(seed).integers(
                0, num_observations, size=(num_steps, batch_size), dtype=np.int64
            )
        )
    batch_indices = np.stack(batch_plans)
    noise_1 = np.stack(
        [np.random.default_rng(seed).standard_normal((num_steps, batch_size)) for seed in seeds_noise_1]
    )
    noise_2 = np.stack(
        [np.random.default_rng(seed).standard_normal((num_steps, batch_size)) for seed in seeds_noise_2]
    )
    if initial_std == 0.0:
        initial_1 = np.zeros((num_replicas, 3), dtype=np.float64)
        initial_2 = np.zeros((num_replicas, 3), dtype=np.float64)
    else:
        if seeds_initialization is None or len(seeds_initialization) != num_replicas:
            raise ValueError("La ablación gaussiana requiere una semilla por réplica")
        initial_1 = np.stack(
            [np.random.default_rng(seed).normal(0.0, initial_std, 3) for seed in seeds_initialization]
        )
        initial_2 = np.stack(
            [np.random.default_rng(seed + 1).normal(0.0, initial_std, 3) for seed in seeds_initialization]
        )
    theta_1 = history_1 = theta_2 = history_2 = None
    if train_block_1:
        theta_1, history_1 = _fit_block_batched(
            actions,
            rewards,
            1,
            rho,
            loss,
            learning_rate,
            batch_indices,
            noise_1,
            initial_1,
            record_every,
        )
    if train_block_2:
        theta_2, history_2 = _fit_block_batched(
            actions,
            rewards,
            2,
            rho,
            loss,
            learning_rate,
            batch_indices,
            noise_2,
            initial_2,
            record_every,
        )
    return FitResult(
        theta_1,
        theta_2,
        history_1,
        history_2,
        num_steps if train_block_1 else 0,
        num_steps if train_block_2 else 0,
    )
