from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from noprop_rl.model import (
    LogisticBlock,
    compose_mc,
    corrupt_rewards,
    ideal_parameters,
    stable_sigmoid,
)
from noprop_rl.quadrature import Quadrature
from noprop_rl.training import fit_blocks_batched


def test_exactly_six_disjoint_parameters() -> None:
    first = LogisticBlock()
    second = LogisticBlock()
    assert first.parameters.shape == second.parameters.shape == (3,)
    assert not np.shares_memory(first.parameters, second.parameters)


def test_corruptions_match_equation_23() -> None:
    rewards = np.array([0.0, 1.0])
    noise_1 = np.array([0.25, -0.75])
    noise_2 = np.array([1.0, -2.0])
    z1, z2 = corrupt_rewards(rewards, 0.5, noise_1, noise_2)
    assert np.array_equal(z1, noise_1)
    assert np.allclose(z2, np.sqrt(0.5) * rewards + np.sqrt(0.5) * noise_2)


def test_mse_and_bce_gradients_match_central_differences() -> None:
    parameters = np.array([0.3, -0.2, 0.4])
    z = np.array([-1.0, 0.2, 0.8, 1.3])
    actions = np.array([0, 1, 0, 1])
    rewards = np.array([0.0, 1.0, 1.0, 0.0])
    for loss in ("mse", "bce"):
        _, analytical = LogisticBlock(parameters).loss_and_gradient(z, actions, rewards, loss)
        numerical = []
        for coordinate in range(3):
            plus = parameters.copy()
            minus = parameters.copy()
            plus[coordinate] += 1e-6
            minus[coordinate] -= 1e-6
            plus_loss = LogisticBlock(plus).loss_and_gradient(z, actions, rewards, loss)[0]
            minus_loss = LogisticBlock(minus).loss_and_gradient(z, actions, rewards, loss)[0]
            numerical.append((plus_loss - minus_loss) / 2e-6)
        assert np.allclose(analytical, numerical, atol=1e-8, rtol=1e-8)


class FakeRng:
    def __init__(self) -> None:
        self.normal_calls = 0
        self.binomial_calls = 0

    def standard_normal(self, size: int) -> np.ndarray:
        self.normal_calls += 1
        return np.zeros(size)

    def binomial(self, _n: int, probabilities: np.ndarray) -> np.ndarray:
        self.binomial_calls += 1
        return np.arange(len(probabilities)) % 2


def test_central_sampler_uses_bernoulli_intermediate_and_averages_p2() -> None:
    first = LogisticBlock(np.zeros(3))
    second = LogisticBlock(np.array([1.0, 0.0, 0.0]))
    rng = FakeRng()
    value = compose_mc(first, second, 0, 0.5, 4, rng)
    expected = 0.5 * (0.5 + float(stable_sigmoid(np.sqrt(0.5))))
    assert np.isclose(value, expected)
    assert rng.binomial_calls == 1
    assert rng.normal_calls == 2

    ablation_rng = FakeRng()
    ablation = compose_mc(first, second, 0, 0.5, 4, ablation_rng, mean_substitute=True)
    assert np.isclose(ablation, stable_sigmoid(np.sqrt(0.5) * 0.5))
    assert ablation_rng.binomial_calls == 0


def test_ideal_composition_and_direct_readouts() -> None:
    means = (0.3, 0.7)
    theta_1, theta_2 = ideal_parameters(means, 0.5)
    quadrature = Quadrature(160)
    for action, mean_reward in enumerate(means):
        assert abs(quadrature.composition_value(theta_1, theta_2, action, 0.5) - mean_reward) < 1e-10
        assert abs(quadrature.d1_value(theta_1, action) - mean_reward) < 1e-12
        assert abs(quadrature.d2_value(theta_2, action, 0.5) - mean_reward) < 1e-12


def test_batched_replicas_equal_separate_fits() -> None:
    actions = np.tile(np.array([0, 0, 1, 1, 0, 1]), (2, 1))
    rewards = np.array([[0, 1, 1, 1, 0, 1], [1, 0, 0, 1, 1, 0]], dtype=np.float64)
    seeds_batch = [101, 202]
    seeds_noise_1 = [303, 404]
    seeds_noise_2 = [505, 606]
    together = fit_blocks_batched(
        actions,
        rewards,
        0.5,
        "mse",
        0.1,
        20,
        4,
        seeds_batch,
        seeds_noise_1,
        seeds_noise_2,
        record_every=20,
    )
    for replica in range(2):
        separate = fit_blocks_batched(
            actions[replica : replica + 1],
            rewards[replica : replica + 1],
            0.5,
            "mse",
            0.1,
            20,
            4,
            [seeds_batch[replica]],
            [seeds_noise_1[replica]],
            [seeds_noise_2[replica]],
            record_every=20,
        )
        assert np.array_equal(together.theta_1[replica], separate.theta_1[0])
        assert np.array_equal(together.theta_2[replica], separate.theta_2[0])
