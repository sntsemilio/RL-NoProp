"""Expectativas gaussianas independientes del sampler desplegado."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np

from .model import ideal_parameters, stable_sigmoid


@dataclass(frozen=True)
class Quadrature:
    order: int = 80

    def __post_init__(self) -> None:
        nodes, weights = np.polynomial.hermite.hermgauss(self.order)
        object.__setattr__(self, "normal_nodes", np.sqrt(2.0) * nodes)
        object.__setattr__(self, "normal_weights", weights / np.sqrt(np.pi))

    def normal_expectation(self, function, mean: float = 0.0, std: float = 1.0) -> float:
        values = function(mean + std * self.normal_nodes)
        return float(np.dot(self.normal_weights, values))

    def block_expectation(self, theta: np.ndarray, action: int, mean: float = 0.0, std: float = 1.0) -> float:
        return self.normal_expectation(
            lambda z: stable_sigmoid(theta[0] * z + theta[1 + action]), mean, std
        )

    def d1_value(self, theta_1: np.ndarray, action: int) -> float:
        return self.block_expectation(theta_1, action)

    def d2_value(self, theta_2: np.ndarray, action: int, rho: float) -> float:
        return float(stable_sigmoid(theta_2[0] * (np.sqrt(rho) / 2.0) + theta_2[1 + action]))

    def composition_value(self, theta_1: np.ndarray, theta_2: np.ndarray, action: int, rho: float) -> float:
        qbar = self.d1_value(theta_1, action)
        std = np.sqrt(1.0 - rho)
        g0 = self.block_expectation(theta_2, action, 0.0, std)
        g1 = self.block_expectation(theta_2, action, np.sqrt(rho), std)
        return (1.0 - qbar) * g0 + qbar * g1

    def mean_substitute_value(
        self, theta_1: np.ndarray, theta_2: np.ndarray, action: int, rho: float
    ) -> float:
        # Expectativa anidada en Z1 y el ruido de Z2; no se usa para el método central.
        p1 = stable_sigmoid(theta_1[0] * self.normal_nodes + theta_1[1 + action])
        inner = []
        for probability in p1:
            inner.append(
                self.block_expectation(
                    theta_2,
                    action,
                    np.sqrt(rho) * float(probability),
                    np.sqrt(1.0 - rho),
                )
            )
        return float(np.dot(self.normal_weights, np.asarray(inner)))

    def posterior_bayes(self, z: np.ndarray, mean_reward: float, rho: float) -> np.ndarray:
        variance = 1.0 - rho
        q0 = np.exp(-(z**2) / (2.0 * variance)) / np.sqrt(2.0 * np.pi * variance)
        q1 = np.exp(-((z - np.sqrt(rho)) ** 2) / (2.0 * variance)) / np.sqrt(
            2.0 * np.pi * variance
        )
        return mean_reward * q1 / ((1.0 - mean_reward) * q0 + mean_reward * q1)

    def local_diagnostics(
        self,
        theta: np.ndarray,
        block: int,
        action: int,
        mean_reward: float,
        rho: float,
    ) -> dict[str, float]:
        ideal_1, ideal_2 = ideal_parameters((mean_reward, mean_reward), rho)
        ideal = ideal_1 if block == 1 else ideal_2
        if block == 1:
            shifts = ((1.0 - mean_reward, 0.0, 0.0), (mean_reward, 0.0, 1.0))
            std = 1.0
        elif block == 2:
            shifts = ((1.0 - mean_reward, 0.0, 0.0), (mean_reward, np.sqrt(rho), 1.0))
            std = np.sqrt(1.0 - rho)
        else:
            raise ValueError("block debe ser 1 o 2")
        mse = 0.0
        bce = 0.0
        excess = 0.0
        l1_error = 0.0
        for probability, shift, label in shifts:
            logits = theta[0] * (shift + std * self.normal_nodes) + theta[1 + action]
            predictions = stable_sigmoid(logits)
            ideal_predictions = stable_sigmoid(
                ideal[0] * (shift + std * self.normal_nodes) + ideal[1 + action]
            )
            mse += probability * 0.5 * float(
                np.dot(self.normal_weights, (predictions - label) ** 2)
            )
            bce += probability * float(
                np.dot(self.normal_weights, np.logaddexp(0.0, logits) - label * logits)
            )
            excess += probability * 0.5 * float(
                np.dot(self.normal_weights, (predictions - ideal_predictions) ** 2)
            )
            l1_error += probability * float(
                np.dot(self.normal_weights, np.abs(predictions - ideal_predictions))
            )
        return {"mse": mse, "bce": bce, "excess": excess, "l1_error": l1_error}


def total_variation_gaussians(rho: float) -> float:
    argument = np.sqrt(rho) / (2.0 * np.sqrt(1.0 - rho))
    normal_cdf = 0.5 * (1.0 + erf(argument / sqrt(2.0)))
    return 2.0 * normal_cdf - 1.0
