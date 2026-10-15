#!/usr/bin/env python3
"""Ejecuta la puerta E0 completa de la sección 7.3."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from noprop_rl.io_utils import RunLedger, atomic_write_json, runtime_metadata, write_csv
from noprop_rl.model import LogisticBlock, ideal_parameters, stable_sigmoid
from noprop_rl.protocol import E0_RHOS, INSTANCES, derived_seed, instance_id
from noprop_rl.quadrature import Quadrature, total_variation_gaussians


def finite_difference_gradient(
    parameters: np.ndarray,
    z: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    loss: str,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    analytical = LogisticBlock(parameters).loss_and_gradient(z, actions, rewards, loss)[1]
    numerical = np.empty(3, dtype=np.float64)
    for coordinate in range(3):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[coordinate] += epsilon
        minus[coordinate] -= epsilon
        loss_plus = LogisticBlock(plus).loss_and_gradient(z, actions, rewards, loss)[0]
        loss_minus = LogisticBlock(minus).loss_and_gradient(z, actions, rewards, loss)[0]
        numerical[coordinate] = (loss_plus - loss_minus) / (2.0 * epsilon)
    return analytical, numerical


def locality_checks() -> list[dict[str, object]]:
    actions = np.array([0, 1, 0, 1, 1, 0], dtype=np.int64)
    rewards = np.array([1, 0, 0, 1, 1, 0], dtype=np.float64)
    z1 = np.array([-0.7, 0.2, 1.1, -1.4, 0.8, 0.1], dtype=np.float64)
    z2 = np.array([0.4, -0.9, 1.3, 0.0, -0.2, 0.6], dtype=np.float64)
    rows: list[dict[str, object]] = []
    for target in (1, 2):
        block_1 = LogisticBlock(np.array([0.2, -0.1, 0.4]))
        block_2 = LogisticBlock(np.array([-0.3, 0.5, -0.2]))
        own = block_1 if target == 1 else block_2
        other = block_2 if target == 1 else block_1
        z = z1 if target == 1 else z2
        other_parameters_before = other.parameters.copy()
        other_state_before = (other.evaluation_count, other.update_count)
        own_parameters_before = own.parameters.copy()
        loss_before = own.loss_and_gradient(z, actions, rewards, "mse")[0]
        calls_to_other_during_loss = other.evaluation_count - other_state_before[0]
        other.parameters += np.array([0.91, -0.73, 0.62])
        loss_after_other_perturbation = own.loss_and_gradient(z, actions, rewards, "mse")[0]
        perturbed_other = other.parameters.copy()
        other_state_pre_update = (other.evaluation_count, other.update_count)
        own.update(z, actions, rewards, "mse", 0.01)
        other_state_post_update = (other.evaluation_count, other.update_count)
        rows.append(
            {
                "target_block": target,
                "loss_invariance_abs_error": abs(loss_before - loss_after_other_perturbation),
                "calls_to_other_during_loss": calls_to_other_during_loss,
                "calls_to_other_during_update": other_state_post_update[0] - other_state_pre_update[0],
                "updates_to_other_during_update": other_state_post_update[1] - other_state_pre_update[1],
                "other_parameters_unchanged_by_target_update": bool(
                    np.array_equal(other.parameters, perturbed_other)
                ),
                "target_parameters_changed": bool(not np.array_equal(own.parameters, own_parameters_before)),
                "other_was_deliberately_perturbed": bool(
                    not np.array_equal(other_parameters_before, perturbed_other)
                ),
            }
        )
    return rows


def p2_moments(
    quadrature: Quadrature,
    theta_2: np.ndarray,
    action: int,
    rho: float,
    mixture_probability: float,
) -> tuple[float, float]:
    nodes = quadrature.normal_nodes
    weights = quadrature.normal_weights
    std = np.sqrt(1.0 - rho)
    logits_0 = theta_2[0] * (std * nodes) + theta_2[1 + action]
    logits_1 = theta_2[0] * (np.sqrt(rho) + std * nodes) + theta_2[1 + action]
    values_0 = stable_sigmoid(logits_0)
    values_1 = stable_sigmoid(logits_1)
    mean = (1.0 - mixture_probability) * np.dot(weights, values_0) + mixture_probability * np.dot(
        weights, values_1
    )
    second = (1.0 - mixture_probability) * np.dot(weights, values_0**2) + mixture_probability * np.dot(
        weights, values_1**2
    )
    return float(mean), float(second)


def run(output_directory: Path) -> dict[str, object]:
    start = time.perf_counter()
    ledger = RunLedger(output_directory / "run_ledger.jsonl")
    ledger.append({"experiment": "E0", "status": "started"})
    quadrature_80 = Quadrature(80)
    quadrature_160 = Quadrature(160)
    analytic_rows: list[dict[str, object]] = []
    bound_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    mc_rows: list[dict[str, object]] = []

    grid = np.linspace(-3.0, 4.0, 701)
    for instance_index, means in enumerate(INSTANCES):
        for rho_index, rho in enumerate(E0_RHOS):
            theta_1, theta_2 = ideal_parameters(means, rho)
            for action, mean_reward in enumerate(means):
                bayes = quadrature_160.posterior_bayes(grid, mean_reward, rho)
                logistic = stable_sigmoid(theta_2[0] * grid + theta_2[1 + action])
                composition_80 = quadrature_80.composition_value(theta_1, theta_2, action, rho)
                composition_160 = quadrature_160.composition_value(theta_1, theta_2, action, rho)
                integral_d2 = quadrature_160.composition_value(theta_1, theta_2, action, rho)
                d1_direct = quadrature_160.d1_value(theta_1, action)
                d2_direct = quadrature_160.d2_value(theta_2, action, rho)
                analytic_rows.append(
                    {
                        "instance": instance_id(means),
                        "rho": rho,
                        "arm": action + 1,
                        "true_mean": mean_reward,
                        "posterior_max_abs_error": float(np.max(np.abs(bayes - logistic))),
                        "ideal_composition": composition_160,
                        "ideal_composition_abs_error": abs(composition_160 - mean_reward),
                        "integral_d2_pa_abs_error": abs(integral_d2 - mean_reward),
                        "d1_direct_abs_error": abs(d1_direct - mean_reward),
                        "d2_direct_abs_error": abs(d2_direct - mean_reward),
                        "quadrature_stability": abs(composition_160 - composition_80),
                    }
                )

                perturbations = {
                    "d1_only": (np.array([0.13, -0.17, 0.11]), np.zeros(3)),
                    "d2_only": (np.zeros(3), np.array([-0.09, 0.14, -0.12])),
                    "both": (
                        np.array([0.13, -0.17, 0.11]),
                        np.array([-0.09, 0.14, -0.12]),
                    ),
                }
                for scenario, (delta_1, delta_2) in perturbations.items():
                    learned_1 = theta_1 + delta_1
                    learned_2 = theta_2 + delta_2
                    value = quadrature_160.composition_value(learned_1, learned_2, action, rho)
                    mean_substitute_perturbed = quadrature_160.mean_substitute_value(
                        learned_1, learned_2, action, rho
                    )
                    qbar = quadrature_160.d1_value(learned_1, action)
                    diagnostic_1 = quadrature_160.local_diagnostics(
                        learned_1, 1, action, mean_reward, rho
                    )
                    diagnostic_2 = quadrature_160.local_diagnostics(
                        learned_2, 2, action, mean_reward, rho
                    )
                    d1 = abs(qbar - mean_reward)
                    d2 = diagnostic_2["l1_error"]
                    bound_45 = total_variation_gaussians(rho) * d1 + d2
                    bound_46 = total_variation_gaussians(rho) * np.sqrt(
                        2.0 * diagnostic_1["excess"]
                    ) + np.sqrt(2.0 * diagnostic_2["excess"])
                    error = abs(value - mean_reward)
                    bound_rows.append(
                        {
                            "instance": instance_id(means),
                            "rho": rho,
                            "arm": action + 1,
                            "scenario": scenario,
                            "value_error": error,
                            "mean_substitute_value_error": abs(
                                mean_substitute_perturbed - mean_reward
                            ),
                            "mean_substitute_minus_bernoulli_gaussian": mean_substitute_perturbed
                            - value,
                            "bounds_apply_to_mean_substitute": False,
                            "d1": d1,
                            "d2": d2,
                            "excess_risk_1": diagnostic_1["excess"],
                            "excess_risk_2": diagnostic_2["excess"],
                            "bound_45": bound_45,
                            "slack_45": bound_45 - error,
                            "bound_46": bound_46,
                            "slack_46": bound_46 - error,
                        }
                    )

                mean_substitute = quadrature_160.mean_substitute_value(theta_1, theta_2, action, rho)
                bernoulli_composition = quadrature_160.composition_value(theta_1, theta_2, action, rho)
                transition_rows.append(
                    {
                        "instance": instance_id(means),
                        "rho": rho,
                        "arm": action + 1,
                        "true_mean": mean_reward,
                        "bernoulli_gaussian_value": bernoulli_composition,
                        "mean_substitute_value": mean_substitute,
                        "absolute_value_difference": abs(mean_substitute - bernoulli_composition),
                        "bernoulli_gaussian_input_variance": (1.0 - rho)
                        + rho * mean_reward * (1.0 - mean_reward),
                        "mean_substitute_input_variance": 1.0 - rho,
                    }
                )

                block_1 = LogisticBlock(theta_1)
                block_2 = LogisticBlock(theta_2)
                reference_mean, reference_second = p2_moments(
                    quadrature_160, theta_2, action, rho, mean_reward
                )
                variance_p2 = max(0.0, reference_second - reference_mean**2)
                for num_replicas in (1, 8, 32, 128, 512):
                    rng = np.random.default_rng(
                        derived_seed(11000, instance_index, rho_index, action, num_replicas)
                    )
                    repeated_actions = np.full((2000, num_replicas), action, dtype=np.int64)
                    z1 = rng.standard_normal((2000, num_replicas))
                    p1 = block_1.predict(z1.ravel(), repeated_actions.ravel()).reshape(z1.shape)
                    intermediate = rng.binomial(1, p1)
                    z2 = np.sqrt(rho) * intermediate + np.sqrt(1.0 - rho) * rng.standard_normal(
                        (2000, num_replicas)
                    )
                    p2 = block_2.predict(z2.ravel(), repeated_actions.ravel()).reshape(z2.shape)
                    estimates = np.mean(p2, axis=1)
                    final_estimates = np.mean(rng.binomial(1, p2), axis=1)
                    sample_variance = float(np.var(estimates, ddof=1))
                    theoretical_variance = variance_p2 / num_replicas
                    standard_error = float(np.std(estimates, ddof=1) / np.sqrt(2000))
                    bias = float(np.mean(estimates) - reference_mean)
                    variance_standard_error = theoretical_variance * np.sqrt(2.0 / 1999.0)
                    mc_rows.append(
                        {
                            "instance": instance_id(means),
                            "rho": rho,
                            "arm": action + 1,
                            "M": num_replicas,
                            "repetitions": 2000,
                            "reference_mean": reference_mean,
                            "estimate_mean": float(np.mean(estimates)),
                            "bias": bias,
                            "standard_error_of_mean": standard_error,
                            "bias_ci95_low": bias - 1.96 * standard_error,
                            "bias_ci95_high": bias + 1.96 * standard_error,
                            "absolute_bias_z": abs(bias) / max(standard_error, np.finfo(float).tiny),
                            "sample_variance": sample_variance,
                            "theoretical_variance": theoretical_variance,
                            "variance_z": abs(sample_variance - theoretical_variance)
                            / max(variance_standard_error, np.finfo(float).tiny),
                            "final_sample_variance": float(np.var(final_estimates, ddof=1)),
                            "final_theoretical_variance": mean_reward
                            * (1.0 - mean_reward)
                            / num_replicas,
                        }
                    )

    gradient_rows: list[dict[str, object]] = []
    rng_gradient = np.random.default_rng(12000)
    actions = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.int64)
    rewards = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.float64)
    for block in (1, 2):
        z = rng_gradient.normal(0.15 * block, 0.9, len(actions))
        parameters = np.array([0.27, -0.31, 0.44]) + 0.05 * block
        for loss in ("mse", "bce"):
            analytical, numerical = finite_difference_gradient(parameters, z, actions, rewards, loss)
            for coordinate in range(3):
                denominator = max(1.0, abs(analytical[coordinate]), abs(numerical[coordinate]))
                gradient_rows.append(
                    {
                        "block": block,
                        "loss": loss,
                        "coordinate": coordinate,
                        "analytical": analytical[coordinate],
                        "numerical": numerical[coordinate],
                        "normalized_error": abs(analytical[coordinate] - numerical[coordinate])
                        / denominator,
                        "epsilon": 1e-6,
                    }
                )
    locality_rows = locality_checks()

    checks = {
        "posterior": max(float(row["posterior_max_abs_error"]) for row in analytic_rows) < 1e-6,
        "ideal_composition": max(float(row["ideal_composition_abs_error"]) for row in analytic_rows)
        < 1e-6,
        "direct_readouts": max(
            max(float(row["d1_direct_abs_error"]), float(row["d2_direct_abs_error"]))
            for row in analytic_rows
        )
        < 1e-6,
        "quadrature_stability": max(float(row["quadrature_stability"]) for row in analytic_rows)
        < 1e-8,
        "gradients": max(float(row["normalized_error"]) for row in gradient_rows) < 1e-5,
        "locality": all(
            float(row["loss_invariance_abs_error"]) == 0.0
            and int(row["calls_to_other_during_loss"]) == 0
            and int(row["calls_to_other_during_update"]) == 0
            and int(row["updates_to_other_during_update"]) == 0
            and bool(row["other_parameters_unchanged_by_target_update"])
            and bool(row["target_parameters_changed"])
            for row in locality_rows
        ),
        "bound_45": min(float(row["slack_45"]) for row in bound_rows) >= -1e-12,
        "bound_46": min(float(row["slack_46"]) for row in bound_rows) >= -1e-12,
        # Criterios MC operativos: seis errores estándar; no son tolerancias matemáticas del PDF.
        "mc_mean_consistent_with_uncertainty": max(float(row["absolute_bias_z"]) for row in mc_rows)
        < 6.0,
        "mc_variance_consistent_with_uncertainty": max(float(row["variance_z"]) for row in mc_rows)
        < 6.0,
        "transitions_distinct": max(
            float(row["absolute_value_difference"]) for row in transition_rows
        )
        > 1e-8,
    }
    summary = {
        "experiment": "E0",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "maxima": {
            "posterior_abs_error": max(float(row["posterior_max_abs_error"]) for row in analytic_rows),
            "ideal_abs_error": max(float(row["ideal_composition_abs_error"]) for row in analytic_rows),
            "quadrature_stability": max(float(row["quadrature_stability"]) for row in analytic_rows),
            "gradient_normalized_error": max(float(row["normalized_error"]) for row in gradient_rows),
            "mc_bias_z": max(float(row["absolute_bias_z"]) for row in mc_rows),
            "mc_variance_z": max(float(row["variance_z"]) for row in mc_rows),
        },
        "minima": {
            "bound_45_slack": min(float(row["slack_45"]) for row in bound_rows),
            "bound_46_slack": min(float(row["slack_46"]) for row in bound_rows),
        },
        "duration_seconds": time.perf_counter() - start,
        "runtime": runtime_metadata(PROJECT),
    }
    write_csv(output_directory / "analytic_checks.csv", analytic_rows)
    write_csv(output_directory / "gradient_checks.csv", gradient_rows)
    write_csv(output_directory / "locality_checks.csv", locality_rows)
    write_csv(output_directory / "bounds_checks.csv", bound_rows)
    write_csv(output_directory / "transition_checks.csv", transition_rows)
    write_csv(output_directory / "monte_carlo_checks.csv", mc_rows)
    atomic_write_json(output_directory / "summary.json", summary)
    ledger.append({"experiment": "E0", "status": summary["status"], "duration_seconds": summary["duration_seconds"]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "results" / "raw" / "e0", help="Directorio de salida"
    )
    args = parser.parse_args()
    summary = run(args.output)
    print(f"E0: {summary['status']} ({summary['duration_seconds']:.2f} s)")
    print(summary["maxima"])
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
