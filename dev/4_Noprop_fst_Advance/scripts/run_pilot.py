#!/usr/bin/env python3
"""Piloto prefijado que congela una tasa por familia antes de E1."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from noprop_rl.io_utils import (
    RunLedger,
    atomic_write_json,
    runtime_metadata,
    utc_now,
    write_csv,
)
from noprop_rl.protocol import (
    LEARNING_RATE_CANDIDATES,
    PILOT_INSTANCES,
    PILOT_SEEDS,
    derived_seed,
    instance_id,
)
from noprop_rl.quadrature import Quadrature
from noprop_rl.training import fit_blocks_batched

PILOT_N = 512  # Decisión operativa congelada antes de observar resultados.
VALIDATION_N = 4096
NUM_STEPS = 2000
BATCH_SIZE = 64
RHO = 0.5


def generate_data(means: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = np.tile(np.repeat(np.arange(2), PILOT_N), (len(PILOT_SEEDS), 1))
    rewards = []
    validation_means = []
    for seed in PILOT_SEEDS:
        training_parts = []
        validation_parts = []
        for action, mean_reward in enumerate(means):
            training_rng = np.random.default_rng(derived_seed(seed, 1, action))
            validation_rng = np.random.default_rng(derived_seed(seed, 2, action))
            training_parts.append(training_rng.binomial(1, mean_reward, PILOT_N))
            validation_parts.append(float(np.mean(validation_rng.binomial(1, mean_reward, VALIDATION_N))))
        rewards.append(np.concatenate(training_parts))
        validation_means.append(validation_parts)
    return actions, np.asarray(rewards, dtype=np.float64), np.asarray(validation_means)


def run_candidate(arguments: tuple[str, float]) -> list[dict[str, object]]:
    loss, learning_rate = arguments
    quadrature = Quadrature(80)
    rows: list[dict[str, object]] = []
    for instance_index, means in enumerate(PILOT_INSTANCES):
        actions, rewards, validation_means = generate_data(means)
        fit = fit_blocks_batched(
            actions,
            rewards,
            rho=RHO,
            loss=loss,
            learning_rate=learning_rate,
            num_steps=NUM_STEPS,
            batch_size=BATCH_SIZE,
            seeds_batch=[derived_seed(seed, 3, instance_index) for seed in PILOT_SEEDS],
            seeds_noise_1=[derived_seed(seed, 4, instance_index) for seed in PILOT_SEEDS],
            seeds_noise_2=[derived_seed(seed, 5, instance_index) for seed in PILOT_SEEDS],
            record_every=100,
        )
        assert fit.theta_1 is not None and fit.theta_2 is not None
        for replica_index, seed in enumerate(PILOT_SEEDS):
            values = [
                quadrature.composition_value(
                    fit.theta_1[replica_index], fit.theta_2[replica_index], action, RHO
                )
                for action in range(2)
            ]
            errors = np.abs(np.asarray(values) - validation_means[replica_index])
            rows.append(
                {
                    "loss": loss,
                    "learning_rate": learning_rate,
                    "instance": instance_id(means),
                    "seed": seed,
                    "training_observations_per_arm": PILOT_N,
                    "validation_rewards_per_arm": VALIDATION_N,
                    "steps": NUM_STEPS,
                    "batch_size": BATCH_SIZE,
                    "validation_mean_arm_1": validation_means[replica_index, 0],
                    "validation_mean_arm_2": validation_means[replica_index, 1],
                    "value_arm_1": values[0],
                    "value_arm_2": values[1],
                    "selection_error_max": float(np.max(errors)),
                    "theta_1_w": fit.theta_1[replica_index, 0],
                    "theta_1_b1": fit.theta_1[replica_index, 1],
                    "theta_1_b2": fit.theta_1[replica_index, 2],
                    "theta_2_w": fit.theta_2[replica_index, 0],
                    "theta_2_b1": fit.theta_2[replica_index, 1],
                    "theta_2_b2": fit.theta_2[replica_index, 2],
                    "final_training_loss_1": fit.loss_history_1[replica_index, -1, 1],
                    "final_training_loss_2": fit.loss_history_2[replica_index, -1, 1],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=PROJECT / "results" / "raw" / "pilot")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger(args.output / "run_ledger.jsonl")
    ledger.append({"experiment": "pilot", "status": "started"})
    e0_summary_path = PROJECT / "results" / "raw" / "e0" / "summary.json"
    if not e0_summary_path.exists() or '"status": "passed"' not in e0_summary_path.read_text():
        raise RuntimeError("E0 no está aprobado; el piloto no puede ejecutarse")
    start = time.perf_counter()
    candidates = [(loss, rate) for loss in ("mse", "bce") for rate in LEARNING_RATE_CANDIDATES]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        nested_rows = list(executor.map(run_candidate, candidates))
    rows = [row for candidate_rows in nested_rows for row in candidate_rows]
    aggregate_rows: list[dict[str, object]] = []
    selected_rates: dict[str, float] = {}
    for loss in ("mse", "bce"):
        loss_aggregates = []
        for rate in LEARNING_RATE_CANDIDATES:
            values = np.asarray(
                [row["selection_error_max"] for row in rows if row["loss"] == loss and row["learning_rate"] == rate],
                dtype=np.float64,
            )
            aggregate = {
                "loss": loss,
                "learning_rate": rate,
                "num_seed_instance_units": len(values),
                "mean_selection_error_max": float(np.mean(values)),
                "median_selection_error_max": float(np.median(values)),
                "standard_error": float(np.std(values, ddof=1) / np.sqrt(len(values))),
            }
            aggregate_rows.append(aggregate)
            loss_aggregates.append(aggregate)
        selected = min(
            loss_aggregates,
            key=lambda row: (float(row["mean_selection_error_max"]), float(row["learning_rate"])),
        )
        selected_rates[loss] = float(selected["learning_rate"])

    duration = time.perf_counter() - start
    frozen_payload = {
        "status": "frozen_before_e1",
        "frozen_at_utc": utc_now(),
        "criterion": "mean across pilot seeds and instances of maximum arm-wise absolute error against independent 4096-reward validation means",
        "true_means_used_for_selection": False,
        "rates": selected_rates,
        "pilot_training_observations_per_arm": PILOT_N,
        "validation_rewards_per_arm": VALIDATION_N,
        "candidate_rates": LEARNING_RATE_CANDIDATES,
        "pilot_seeds": PILOT_SEEDS,
        "pilot_instances": PILOT_INSTANCES,
        "steps_per_fit": NUM_STEPS,
        "batch_size": BATCH_SIZE,
        "rho": RHO,
        "duration_seconds": duration,
        "workers": args.workers,
        "runtime": runtime_metadata(PROJECT),
    }
    write_csv(args.output / "per_seed.csv", rows)
    write_csv(args.output / "aggregate.csv", aggregate_rows)
    atomic_write_json(PROJECT / "configs" / "frozen_rates.json", frozen_payload)
    atomic_write_json(args.output / "summary.json", frozen_payload)
    ledger.append({"experiment": "pilot", "status": "completed", "duration_seconds": duration, "rates": selected_rates})
    print(f"Piloto completado en {duration:.2f} s; tasas congeladas: {selected_rates}")


if __name__ == "__main__":
    main()
