"""Modelo mínimo de seis parámetros y actualizaciones locales explícitas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

LossName = Literal["mse", "bce"]


def stable_sigmoid(logits: np.ndarray | float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(values)
    nonnegative = values >= 0
    output[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_values = np.exp(values[~nonnegative])
    output[~nonnegative] = exp_values / (1.0 + exp_values)
    return output


def design_matrix(z: np.ndarray, actions: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int64)
    if z.shape != actions.shape:
        raise ValueError("z y actions deben tener la misma forma")
    return np.column_stack((z, actions == 0, actions == 1)).astype(np.float64)


@dataclass
class LogisticBlock:
    """Una pendiente y dos sesgos; no contiene referencias al otro bloque."""

    parameters: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    evaluation_count: int = 0
    update_count: int = 0

    def __post_init__(self) -> None:
        self.parameters = np.asarray(self.parameters, dtype=np.float64).copy()
        if self.parameters.shape != (3,):
            raise ValueError("Un bloque debe tener exactamente tres parámetros")

    def logits(self, z: np.ndarray, actions: np.ndarray) -> np.ndarray:
        self.evaluation_count += 1
        x = design_matrix(z, actions)
        return x @ self.parameters

    def predict(self, z: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return stable_sigmoid(self.logits(z, actions))

    def loss_and_gradient(
        self,
        z: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        loss: LossName,
    ) -> tuple[float, np.ndarray]:
        rewards = np.asarray(rewards, dtype=np.float64)
        x = design_matrix(np.asarray(z, dtype=np.float64), actions)
        logits = self.logits(np.asarray(z, dtype=np.float64), actions)
        probabilities = stable_sigmoid(logits)
        if loss == "mse":
            residual = probabilities - rewards
            value = 0.5 * np.mean(residual**2)
            factors = residual * probabilities * (1.0 - probabilities)
        elif loss == "bce":
            value = np.mean(np.logaddexp(0.0, logits) - rewards * logits)
            factors = probabilities - rewards
        else:
            raise ValueError(f"Pérdida desconocida: {loss}")
        return float(value), np.mean(factors[:, None] * x, axis=0)

    def update(
        self,
        z: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        loss: LossName,
        learning_rate: float,
    ) -> float:
        value, gradient = self.loss_and_gradient(z, actions, rewards, loss)
        self.parameters -= learning_rate * gradient
        self.update_count += 1
        return value


def corrupt_rewards(
    rewards: np.ndarray,
    rho: float,
    noise_block_1: np.ndarray,
    noise_block_2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Ecuación (2.3), con ruidos provistos para hacer auditable la localidad."""

    if not 0.0 < rho < 1.0:
        raise ValueError("rho debe pertenecer a (0, 1)")
    rewards = np.asarray(rewards, dtype=np.float64)
    noise_block_1 = np.asarray(noise_block_1, dtype=np.float64)
    noise_block_2 = np.asarray(noise_block_2, dtype=np.float64)
    if rewards.shape != noise_block_1.shape or rewards.shape != noise_block_2.shape:
        raise ValueError("Cada bloque requiere un ruido por recompensa")
    z1 = noise_block_1.copy()
    z2 = np.sqrt(rho) * rewards + np.sqrt(1.0 - rho) * noise_block_2
    return z1, z2


def compose_mc(
    block_1: LogisticBlock,
    block_2: LogisticBlock,
    action: int,
    rho: float,
    num_replicas: int,
    rng: np.random.Generator,
    mean_substitute: bool = False,
    sample_final_reward: bool = False,
) -> float:
    """Ecuación (4.1); la sustitución por la media es solo una ablación explícita."""

    if num_replicas <= 0:
        raise ValueError("num_replicas debe ser positivo")
    actions = np.full(num_replicas, action, dtype=np.int64)
    z1 = rng.standard_normal(num_replicas)
    p1 = block_1.predict(z1, actions)
    if mean_substitute:
        intermediate = p1
    else:
        intermediate = rng.binomial(1, p1).astype(np.float64)
    z2 = np.sqrt(rho) * intermediate + np.sqrt(1.0 - rho) * rng.standard_normal(num_replicas)
    p2 = block_2.predict(z2, actions)
    if sample_final_reward:
        return float(np.mean(rng.binomial(1, p2)))
    return float(np.mean(p2))


def ideal_parameters(means: tuple[float, float], rho: float) -> tuple[np.ndarray, np.ndarray]:
    means_array = np.asarray(means, dtype=np.float64)
    logits = np.log(means_array / (1.0 - means_array))
    theta_1 = np.array([0.0, logits[0], logits[1]], dtype=np.float64)
    theta_2 = np.array(
        [
            np.sqrt(rho) / (1.0 - rho),
            logits[0] - rho / (2.0 * (1.0 - rho)),
            logits[1] - rho / (2.0 * (1.0 - rho)),
        ],
        dtype=np.float64,
    )
    return theta_1, theta_2
