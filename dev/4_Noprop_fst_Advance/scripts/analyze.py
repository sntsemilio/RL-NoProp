#!/usr/bin/env python3
"""Agrega E0/E1/smoke y genera figuras reproducibles."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from noprop_rl.io_utils import atomic_write_json, runtime_metadata, write_csv
from noprop_rl.metrics import bootstrap_interval
from noprop_rl.protocol import BOOTSTRAP_SEED, derived_seed

NUM_BOOTSTRAP = 10_000


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def numeric(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return np.nan


def summarize_values(values: list[float], seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "mean_ci95_low": np.nan,
            "mean_ci95_high": np.nan,
            "median_ci95_low": np.nan,
            "median_ci95_high": np.nan,
        }
    rng_mean = np.random.default_rng(derived_seed(seed, 1))
    rng_median = np.random.default_rng(derived_seed(seed, 2))
    mean_low, mean_high = bootstrap_interval(array, rng_mean, NUM_BOOTSTRAP, "mean")
    median_low, median_high = bootstrap_interval(array, rng_median, NUM_BOOTSTRAP, "median")
    return {
        "n": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "mean_ci95_low": mean_low,
        "mean_ci95_high": mean_high,
        "median_ci95_low": median_low,
        "median_ci95_high": median_high,
    }


def aggregate_e1(rows: list[dict[str, str]], output_directory: Path) -> dict[str, object]:
    per_seed_arm_max: dict[tuple[str, int, str, int], float] = defaultdict(float)
    per_arm_groups: dict[tuple[str, int, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["instance"], int(row["N_per_arm"]), row["method"], int(row["seed"]))
        per_seed_arm_max[key] = max(per_seed_arm_max[key], numeric(row, "abs_value_error"))
        per_arm_groups[
            (row["instance"], int(row["N_per_arm"]), row["method"], int(row["arm"]))
        ].append(row)

    max_groups: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for (instance, sample_size, method, _seed), value in per_seed_arm_max.items():
        max_groups[(instance, sample_size, method)].append(value)
    max_rows = []
    for group_index, ((instance, sample_size, method), values) in enumerate(sorted(max_groups.items())):
        max_rows.append(
            {
                "instance": instance,
                "N_per_arm": sample_size,
                "method": method,
                "metric": "max_arm_abs_value_error",
                **summarize_values(values, derived_seed(BOOTSTRAP_SEED, 10, group_index)),
                "interval_type": "pointwise percentile bootstrap over complete seeds",
                "bootstrap_resamples": NUM_BOOTSTRAP,
            }
        )
    write_csv(output_directory / "e1_primary_by_instance.csv", max_rows)

    metrics = [
        "abs_value_error",
        "value_calibration_abs_error",
        "excess_risk_1",
        "excess_risk_2",
        "population_mse_1",
        "population_mse_2",
        "population_bce_1",
        "population_bce_2",
        "denoiser_ece_1",
        "denoiser_ece_2",
        "bound_45",
        "bound_45_slack",
        "bound_46",
        "bound_46_slack",
        "theta_1_distance_ideal",
        "theta_2_distance_ideal",
    ]
    arm_rows = []
    counter = 0
    for (instance, sample_size, method, arm), group_rows in sorted(per_arm_groups.items()):
        for metric in metrics:
            values = [numeric(row, metric) for row in group_rows]
            if not np.any(np.isfinite(values)):
                continue
            arm_rows.append(
                {
                    "instance": instance,
                    "N_per_arm": sample_size,
                    "method": method,
                    "arm": arm,
                    "metric": metric,
                    **summarize_values(values, derived_seed(BOOTSTRAP_SEED, 20, counter)),
                    "interval_type": "pointwise percentile bootstrap over complete seeds",
                    "bootstrap_resamples": NUM_BOOTSTRAP,
                }
            )
            counter += 1
    write_csv(output_directory / "e1_per_arm_aggregate.csv", arm_rows)

    comparisons = [
        "tabular_mean",
        "d1_independent_mse",
        "d2_independent_mse",
        "mean_substitute_mse",
        "d1_readout_from_two_mse",
        "d2_readout_from_two_mse",
    ]
    paired_rows = []
    for comparison_index, ((instance, sample_size, _method), composition_values) in enumerate(
        sorted(
            (key, values)
            for key, values in max_groups.items()
            if key[2] == "two_block_mse"
        )
    ):
        composition_by_seed = {
            seed: value
            for (row_instance, row_n, method, seed), value in per_seed_arm_max.items()
            if row_instance == instance and row_n == sample_size and method == "two_block_mse"
        }
        for control_index, control in enumerate(comparisons):
            control_by_seed = {
                seed: value
                for (row_instance, row_n, method, seed), value in per_seed_arm_max.items()
                if row_instance == instance and row_n == sample_size and method == control
            }
            common_seeds = sorted(composition_by_seed.keys() & control_by_seed.keys())
            differences = [
                composition_by_seed[seed] - control_by_seed[seed] for seed in common_seeds
            ]
            paired_rows.append(
                {
                    "instance": instance,
                    "N_per_arm": sample_size,
                    "method": "two_block_mse",
                    "control": control,
                    "difference": "composition minus control; negative favors composition",
                    **summarize_values(
                        differences,
                        derived_seed(BOOTSTRAP_SEED, 30, comparison_index, control_index),
                    ),
                    "paired": True,
                    "bootstrap_resamples": NUM_BOOTSTRAP,
                }
            )
    write_csv(output_directory / "e1_paired_differences.csv", paired_rows)

    across_instances: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for (instance, sample_size, method, seed), value in per_seed_arm_max.items():
        across_instances[(sample_size, method, seed)].append(value)
    seed_means = {
        key: float(np.mean(values)) for key, values in across_instances.items() if len(values) == 9
    }
    high_level_groups: dict[tuple[int, str], list[float]] = defaultdict(list)
    for (sample_size, method, _seed), value in seed_means.items():
        high_level_groups[(sample_size, method)].append(value)
    high_level_rows = []
    for group_index, ((sample_size, method), values) in enumerate(sorted(high_level_groups.items())):
        high_level_rows.append(
            {
                "N_per_arm": sample_size,
                "method": method,
                "aggregation": "mean of per-seed maximum arm error across all nine instances",
                **summarize_values(values, derived_seed(BOOTSTRAP_SEED, 40, group_index)),
                "bootstrap_resamples": NUM_BOOTSTRAP,
            }
        )
    write_csv(output_directory / "e1_across_instances.csv", high_level_rows)

    high_level_paired_rows = []
    comparison_sets = {
        "two_block_mse": comparisons,
        "two_block_bce": [
            "tabular_mean",
            "d1_readout_from_two_bce",
            "d2_readout_from_two_bce",
        ],
    }
    for method_index, (composition_method, controls) in enumerate(comparison_sets.items()):
        for sample_size in sorted({key[0] for key in seed_means}):
            for control_index, control in enumerate(controls):
                differences = []
                for seed in sorted({key[2] for key in seed_means if key[0] == sample_size}):
                    composition_key = (sample_size, composition_method, seed)
                    control_key = (sample_size, control, seed)
                    if composition_key in seed_means and control_key in seed_means:
                        differences.append(seed_means[composition_key] - seed_means[control_key])
                high_level_paired_rows.append(
                    {
                        "N_per_arm": sample_size,
                        "method": composition_method,
                        "control": control,
                        "aggregation": "paired difference of each seed's mean maximum error across nine instances",
                        "difference": "composition minus control; negative favors composition",
                        **summarize_values(
                            differences,
                            derived_seed(
                                BOOTSTRAP_SEED,
                                50,
                                method_index,
                                sample_size,
                                control_index,
                            ),
                        ),
                        "paired": True,
                        "bootstrap_resamples": NUM_BOOTSTRAP,
                    }
                )
    write_csv(output_directory / "e1_paired_across_instances.csv", high_level_paired_rows)

    finite_slack_45 = [numeric(row, "bound_45_slack") for row in rows]
    finite_slack_46 = [numeric(row, "bound_46_slack") for row in rows]
    finite_slack_45 = [value for value in finite_slack_45 if np.isfinite(value)]
    finite_slack_46 = [value for value in finite_slack_46 if np.isfinite(value)]
    return {
        "num_raw_rows": len(rows),
        "num_complete_seed_units": len(per_seed_arm_max),
        "min_bound_45_slack": min(finite_slack_45),
        "min_bound_46_slack": min(finite_slack_46),
        "bounds_hold_numerically": min(finite_slack_45 + finite_slack_46) >= -1e-10,
    }


def plot_e1(output_directory: Path) -> None:
    rows = read_csv(output_directory / "e1_across_instances.csv")
    methods = [
        "tabular_mean",
        "d1_independent_mse",
        "d2_independent_mse",
        "two_block_mse",
        "two_block_bce",
        "mean_substitute_mse",
    ]
    labels = {
        "tabular_mean": "Media tabular",
        "d1_independent_mse": "Solo D1",
        "d2_independent_mse": "Solo D2",
        "two_block_mse": "Dos bloques MSE",
        "two_block_bce": "Dos bloques BCE",
        "mean_substitute_mse": "Media sustituta",
    }
    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    for method in methods:
        selected = sorted(
            (row for row in rows if row["method"] == method), key=lambda row: int(row["N_per_arm"])
        )
        x = np.asarray([int(row["N_per_arm"]) for row in selected])
        y = np.asarray([float(row["mean"]) for row in selected])
        low = np.asarray([float(row["mean_ci95_low"]) for row in selected])
        high = np.asarray([float(row["mean_ci95_high"]) for row in selected])
        axis.plot(x, y, marker="o", label=labels[method])
        axis.fill_between(x, low, high, alpha=0.12)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Observaciones independientes por brazo (N)")
    axis.set_ylabel("Máximo error absoluto, promedio sobre instancias")
    axis.set_title("E1: aprendizaje de valor (bandas bootstrap puntuales 95 %)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_directory.parent / "figures" / "e1_value_error.png", dpi=180)
    plt.close(fig)


def plot_transfer(raw_rows: list[dict[str, str]], output_directory: Path) -> None:
    selected = [row for row in raw_rows if row["method"] == "two_block_mse"]
    bound = np.asarray([numeric(row, "bound_46") for row in selected])
    error = np.asarray([numeric(row, "abs_value_error") for row in selected])
    fig, axis = plt.subplots(figsize=(5.5, 5.0))
    axis.scatter(bound, error, s=7, alpha=0.25)
    maximum = float(max(np.max(bound), np.max(error)))
    axis.plot([0, maximum], [0, maximum], color="black", linestyle="--", label="error = cota")
    axis.set_xlabel("Cota local (4.6)")
    axis.set_ylabel("Error absoluto de valor")
    axis.set_title("E1: transferencia de error local a valor")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_directory.parent / "figures" / "e1_local_bound.png", dpi=180)
    plt.close(fig)


def plot_e0(project: Path) -> None:
    rows = read_csv(project / "results" / "raw" / "e0" / "monte_carlo_checks.csv")
    groups: dict[int, list[float]] = defaultdict(list)
    theoretical: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        groups[int(row["M"])].append(float(row["sample_variance"]))
        theoretical[int(row["M"])].append(float(row["theoretical_variance"]))
    x = np.asarray(sorted(groups))
    observed_y = np.asarray([np.mean(groups[value]) for value in x])
    theoretical_y = np.asarray([np.mean(theoretical[value]) for value in x])
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.loglog(x, observed_y, marker="o", label="Varianza observada")
    axis.loglog(x, theoretical_y, marker="x", linestyle="--", label="Varianza por cuadratura / M")
    axis.set_xlabel("Réplicas M")
    axis.set_ylabel("Varianza del estimador")
    axis.set_title("E0: M reduce error del sampler, no del entrenamiento")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    fig.tight_layout()
    fig.savefig(project / "results" / "figures" / "e0_sampler_variance.png", dpi=180)
    plt.close(fig)


def plot_benchmark(project: Path) -> None:
    payload = json.loads((project / "results" / "hardware_benchmark.json").read_text())
    names = ["CPU vectorizada", "CPU 4 procesos", "GPU por lotes"]
    values = [
        payload["cpu_vectorized"]["throughput_replicas_per_second"],
        payload["cpu_processes_4"]["throughput_replicas_per_second"],
        payload["gpu"].get("throughput_replicas_per_second", 0.0),
    ]
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.bar(names, values)
    axis.set_ylabel("Réplicas / segundo")
    axis.set_title("Microbenchmark de throughput (32 réplicas)")
    axis.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(project / "results" / "figures" / "hardware_throughput.png", dpi=180)
    plt.close(fig)


def main() -> None:
    raw_path = PROJECT / "results" / "raw" / "e1" / "per_arm.csv"
    output_directory = PROJECT / "results" / "aggregated"
    figures_directory = PROJECT / "results" / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)
    figures_directory.mkdir(parents=True, exist_ok=True)
    rows = read_csv(raw_path)
    e1_summary = aggregate_e1(rows, output_directory)
    plot_e1(output_directory)
    plot_transfer(rows, output_directory)
    plot_e0(PROJECT)
    plot_benchmark(PROJECT)
    summary = {
        "status": "completed",
        "bootstrap_resamples": NUM_BOOTSTRAP,
        "intervals": "pointwise 95% percentile bootstrap; not simultaneous",
        "experimental_unit": "complete seed",
        "paired_comparisons": True,
        "e1": e1_summary,
        "runtime": runtime_metadata(PROJECT),
    }
    atomic_write_json(output_directory / "analysis_summary.json", summary)
    print(json.dumps(summary["e1"], indent=2))


if __name__ == "__main__":
    main()
