#!/usr/bin/env python3
"""E1 confirmatorio: datos prefijados, 30 semillas y controles del Cuadro 1."""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from noprop_rl.io_utils import (
    RunLedger,
    atomic_write_json,
    runtime_metadata,
    sha256_file,
    utc_now,
    write_csv,
)
from noprop_rl.metrics import expected_calibration_error
from noprop_rl.model import ideal_parameters, stable_sigmoid
from noprop_rl.protocol import (
    E1_SAMPLE_SIZES,
    E1_SEEDS,
    INSTANCES,
    derived_seed,
    instance_id,
    load_frozen_rates,
)
from noprop_rl.quadrature import Quadrature, total_variation_gaussians
from noprop_rl.training import FitResult, fit_blocks_batched

RHO = 0.5
NUM_STEPS = 2000
BATCH_SIZE = 64
CALIBRATION_SAMPLES = 4096
RECORD_EVERY = 100


def generate_condition_data(
    means: tuple[float, float], num_observations_per_arm: int, instance_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_seeds = len(E1_SEEDS)
    actions = np.tile(np.repeat(np.arange(2), num_observations_per_arm), (num_seeds, 1))
    rewards = []
    validation_means = []
    calibration_rewards = np.empty((num_seeds, 2, CALIBRATION_SAMPLES), dtype=np.float64)
    calibration_noise_1 = np.empty_like(calibration_rewards)
    calibration_noise_2 = np.empty_like(calibration_rewards)
    for replica_index, seed in enumerate(E1_SEEDS):
        training_parts = []
        validation_parts = []
        for action, mean_reward in enumerate(means):
            train_rng = np.random.default_rng(
                derived_seed(seed, 100, instance_index, num_observations_per_arm, action)
            )
            value_validation_rng = np.random.default_rng(
                derived_seed(seed, 101, instance_index, num_observations_per_arm, action)
            )
            joint_rng = np.random.default_rng(
                derived_seed(seed, 102, instance_index, num_observations_per_arm, action)
            )
            training_parts.append(train_rng.binomial(1, mean_reward, num_observations_per_arm))
            validation_parts.append(
                float(np.mean(value_validation_rng.binomial(1, mean_reward, CALIBRATION_SAMPLES)))
            )
            calibration_rewards[replica_index, action] = joint_rng.binomial(
                1, mean_reward, CALIBRATION_SAMPLES
            )
            calibration_noise_1[replica_index, action] = joint_rng.standard_normal(CALIBRATION_SAMPLES)
            calibration_noise_2[replica_index, action] = joint_rng.standard_normal(CALIBRATION_SAMPLES)
        rewards.append(np.concatenate(training_parts))
        validation_means.append(validation_parts)
    return (
        actions,
        np.asarray(rewards, dtype=np.float64),
        np.asarray(validation_means, dtype=np.float64),
        calibration_rewards,
        calibration_noise_1,
        calibration_noise_2,
    )


def timed_fit(*args, **kwargs) -> tuple[FitResult, float]:
    start = time.perf_counter()
    fit = fit_blocks_batched(*args, **kwargs)
    return fit, time.perf_counter() - start


def fit_model(
    actions: np.ndarray,
    rewards: np.ndarray,
    instance_index: int,
    num_observations_per_arm: int,
    source_code: int,
    loss: str,
    learning_rate: float,
    train_block_1: bool = True,
    train_block_2: bool = True,
) -> tuple[FitResult, float]:
    return timed_fit(
        actions,
        rewards,
        rho=RHO,
        loss=loss,
        learning_rate=learning_rate,
        num_steps=NUM_STEPS,
        batch_size=BATCH_SIZE,
        seeds_batch=[
            derived_seed(seed, source_code, 1, instance_index, num_observations_per_arm)
            for seed in E1_SEEDS
        ],
        seeds_noise_1=[
            derived_seed(seed, source_code, 2, instance_index, num_observations_per_arm)
            for seed in E1_SEEDS
        ],
        seeds_noise_2=[
            derived_seed(seed, source_code, 3, instance_index, num_observations_per_arm)
            for seed in E1_SEEDS
        ],
        train_block_1=train_block_1,
        train_block_2=train_block_2,
        initial_std=0.0,
        record_every=RECORD_EVERY,
    )


def calibration_ece(
    theta: np.ndarray,
    block: int,
    action: int,
    rewards: np.ndarray,
    noise_1: np.ndarray,
    noise_2: np.ndarray,
) -> float:
    if block == 1:
        z = noise_1
    else:
        z = np.sqrt(RHO) * rewards + np.sqrt(1.0 - RHO) * noise_2
    predictions = stable_sigmoid(theta[0] * z + theta[1 + action])
    return expected_calibration_error(predictions, rewards, num_bins=10)


def history_rows(source: str, fit: FitResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for block, history in ((1, fit.loss_history_1), (2, fit.loss_history_2)):
        if history is None:
            continue
        for replica_index, seed in enumerate(E1_SEEDS):
            for step, loss_value in history[replica_index]:
                rows.append(
                    {
                        "source_model": source,
                        "seed": seed,
                        "block": block,
                        "step": int(step),
                        "minibatch_training_loss": float(loss_value),
                    }
                )
    return rows


def run_condition(arguments: tuple[int, tuple[float, float], int, dict[str, float], str]) -> dict[str, object]:
    instance_index, means, num_observations_per_arm, rates, checkpoint_directory_string = arguments
    checkpoint_directory = Path(checkpoint_directory_string)
    identifier = f"{instance_id(means)}_N{num_observations_per_arm}"
    result_path = checkpoint_directory / f"{identifier}.csv"
    history_path = checkpoint_directory / f"{identifier}_history.csv"
    metadata_path = checkpoint_directory / f"{identifier}.json"
    start = time.perf_counter()
    try:
        (
            actions,
            rewards,
            validation_means,
            calibration_rewards,
            calibration_noise_1,
            calibration_noise_2,
        ) = generate_condition_data(means, num_observations_per_arm, instance_index)

        composite_mse, duration_composite_mse = fit_model(
            actions, rewards, instance_index, num_observations_per_arm, 200, "mse", rates["mse"]
        )
        composite_bce, duration_composite_bce = fit_model(
            actions, rewards, instance_index, num_observations_per_arm, 300, "bce", rates["bce"]
        )
        independent_d1, duration_independent_d1 = fit_model(
            actions,
            rewards,
            instance_index,
            num_observations_per_arm,
            400,
            "mse",
            rates["mse"],
            train_block_2=False,
        )
        independent_d2, duration_independent_d2 = fit_model(
            actions,
            rewards,
            instance_index,
            num_observations_per_arm,
            500,
            "mse",
            rates["mse"],
            train_block_1=False,
        )
        assert composite_mse.theta_1 is not None and composite_mse.theta_2 is not None
        assert composite_bce.theta_1 is not None and composite_bce.theta_2 is not None
        assert independent_d1.theta_1 is not None and independent_d2.theta_2 is not None

        model_specs = [
            ("tabular_mean", "tabular", None, None, 0.0, "tabular"),
            (
                "d1_independent_mse",
                "independent_d1",
                independent_d1.theta_1,
                None,
                duration_independent_d1,
                "d1_direct",
            ),
            (
                "d2_independent_mse",
                "independent_d2",
                None,
                independent_d2.theta_2,
                duration_independent_d2,
                "d2_direct",
            ),
            (
                "two_block_mse",
                "composite_mse",
                composite_mse.theta_1,
                composite_mse.theta_2,
                duration_composite_mse,
                "bernoulli_gaussian_composition",
            ),
            (
                "two_block_bce",
                "composite_bce",
                composite_bce.theta_1,
                composite_bce.theta_2,
                duration_composite_bce,
                "bernoulli_gaussian_composition",
            ),
            (
                "mean_substitute_mse",
                "composite_mse",
                composite_mse.theta_1,
                composite_mse.theta_2,
                duration_composite_mse,
                "mean_substitute_ablation",
            ),
            (
                "d1_readout_from_two_mse",
                "composite_mse",
                composite_mse.theta_1,
                None,
                duration_composite_mse,
                "d1_direct_same_weights",
            ),
            (
                "d2_readout_from_two_mse",
                "composite_mse",
                None,
                composite_mse.theta_2,
                duration_composite_mse,
                "d2_direct_same_weights",
            ),
            (
                "d1_readout_from_two_bce",
                "composite_bce",
                composite_bce.theta_1,
                None,
                duration_composite_bce,
                "d1_direct_same_weights",
            ),
            (
                "d2_readout_from_two_bce",
                "composite_bce",
                None,
                composite_bce.theta_2,
                duration_composite_bce,
                "d2_direct_same_weights",
            ),
        ]
        fit_by_source = {
            "composite_mse": composite_mse,
            "composite_bce": composite_bce,
            "independent_d1": independent_d1,
            "independent_d2": independent_d2,
        }
        quadrature = Quadrature(80)
        rows: list[dict[str, object]] = []
        tau = total_variation_gaussians(RHO)
        for method, source, theta_1_all, theta_2_all, fit_duration, inference in model_specs:
            for replica_index, seed in enumerate(E1_SEEDS):
                theta_1 = None if theta_1_all is None else theta_1_all[replica_index]
                theta_2 = None if theta_2_all is None else theta_2_all[replica_index]
                for action, mean_reward in enumerate(means):
                    if inference == "tabular":
                        selected_rewards = rewards[replica_index, actions[replica_index] == action]
                        value = float(np.mean(selected_rewards))
                    elif inference.startswith("d1_direct"):
                        value = quadrature.d1_value(theta_1, action)
                    elif inference.startswith("d2_direct"):
                        value = quadrature.d2_value(theta_2, action, RHO)
                    elif inference == "mean_substitute_ablation":
                        value = quadrature.mean_substitute_value(theta_1, theta_2, action, RHO)
                    else:
                        value = quadrature.composition_value(theta_1, theta_2, action, RHO)

                    diagnostics_1 = (
                        quadrature.local_diagnostics(theta_1, 1, action, mean_reward, RHO)
                        if theta_1 is not None
                        else None
                    )
                    diagnostics_2 = (
                        quadrature.local_diagnostics(theta_2, 2, action, mean_reward, RHO)
                        if theta_2 is not None
                        else None
                    )
                    ideal_1, ideal_2 = ideal_parameters(means, RHO)
                    d1 = (
                        abs(quadrature.d1_value(theta_1, action) - mean_reward)
                        if theta_1 is not None
                        else np.nan
                    )
                    d2 = diagnostics_2["l1_error"] if diagnostics_2 is not None else np.nan
                    if (
                        theta_1 is not None
                        and theta_2 is not None
                        and inference == "bernoulli_gaussian_composition"
                    ):
                        bound_45 = tau * d1 + d2
                        bound_46 = tau * np.sqrt(2.0 * diagnostics_1["excess"]) + np.sqrt(
                            2.0 * diagnostics_2["excess"]
                        )
                        slack_45 = bound_45 - abs(value - mean_reward)
                        slack_46 = bound_46 - abs(value - mean_reward)
                    else:
                        bound_45 = bound_46 = slack_45 = slack_46 = np.nan
                    ece_1 = (
                        calibration_ece(
                            theta_1,
                            1,
                            action,
                            calibration_rewards[replica_index, action],
                            calibration_noise_1[replica_index, action],
                            calibration_noise_2[replica_index, action],
                        )
                        if theta_1 is not None
                        else np.nan
                    )
                    ece_2 = (
                        calibration_ece(
                            theta_2,
                            2,
                            action,
                            calibration_rewards[replica_index, action],
                            calibration_noise_1[replica_index, action],
                            calibration_noise_2[replica_index, action],
                        )
                        if theta_2 is not None
                        else np.nan
                    )
                    fit_object = fit_by_source.get(source)
                    source_updates_1 = fit_object.update_count_1 if fit_object is not None else 0
                    source_updates_2 = fit_object.update_count_2 if fit_object is not None else 0
                    final_loss_1 = (
                        fit_object.loss_history_1[replica_index, -1, 1]
                        if fit_object is not None and fit_object.loss_history_1 is not None
                        else np.nan
                    )
                    final_loss_2 = (
                        fit_object.loss_history_2[replica_index, -1, 1]
                        if fit_object is not None and fit_object.loss_history_2 is not None
                        else np.nan
                    )
                    rows.append(
                        {
                            "experiment": "E1",
                            "instance": instance_id(means),
                            "mu1": means[0],
                            "mu2": means[1],
                            "N_per_arm": num_observations_per_arm,
                            "seed": seed,
                            "method": method,
                            "parameter_source": source,
                            "inference": inference,
                            "arm": action + 1,
                            "true_mean": mean_reward,
                            "value_estimate": value,
                            "abs_value_error": abs(value - mean_reward),
                            "validation_mean_4096": validation_means[replica_index, action],
                            "value_calibration_abs_error": abs(
                                value - validation_means[replica_index, action]
                            ),
                            "d1": d1,
                            "d2": d2,
                            "excess_risk_1": diagnostics_1["excess"]
                            if diagnostics_1 is not None
                            else np.nan,
                            "excess_risk_2": diagnostics_2["excess"]
                            if diagnostics_2 is not None
                            else np.nan,
                            "population_mse_1": diagnostics_1["mse"]
                            if diagnostics_1 is not None
                            else np.nan,
                            "population_mse_2": diagnostics_2["mse"]
                            if diagnostics_2 is not None
                            else np.nan,
                            "population_bce_1": diagnostics_1["bce"]
                            if diagnostics_1 is not None
                            else np.nan,
                            "population_bce_2": diagnostics_2["bce"]
                            if diagnostics_2 is not None
                            else np.nan,
                            "denoiser_ece_1": ece_1,
                            "denoiser_ece_2": ece_2,
                            "bound_45": bound_45,
                            "bound_45_slack": slack_45,
                            "bound_46": bound_46,
                            "bound_46_slack": slack_46,
                            "theta_1_distance_ideal": float(np.linalg.norm(theta_1 - ideal_1))
                            if theta_1 is not None
                            else np.nan,
                            "theta_2_distance_ideal": float(np.linalg.norm(theta_2 - ideal_2))
                            if theta_2 is not None
                            else np.nan,
                            "theta_1_w": theta_1[0] if theta_1 is not None else np.nan,
                            "theta_1_b1": theta_1[1] if theta_1 is not None else np.nan,
                            "theta_1_b2": theta_1[2] if theta_1 is not None else np.nan,
                            "theta_2_w": theta_2[0] if theta_2 is not None else np.nan,
                            "theta_2_b1": theta_2[1] if theta_2 is not None else np.nan,
                            "theta_2_b2": theta_2[2] if theta_2 is not None else np.nan,
                            "final_minibatch_loss_1": final_loss_1,
                            "final_minibatch_loss_2": final_loss_2,
                            "fit_duration_seconds_for_source": fit_duration,
                            "training_updates_block_1": source_updates_1,
                            "training_updates_block_2": source_updates_2,
                            "training_sample_evaluations": BATCH_SIZE
                            * (source_updates_1 + source_updates_2),
                            "rho": RHO,
                            "batch_size": BATCH_SIZE,
                            "steps": NUM_STEPS,
                            "initialization": "zeros",
                            "dtype": "float64",
                        }
                    )
        histories: list[dict[str, object]] = []
        for source, fit in fit_by_source.items():
            histories.extend(history_rows(source, fit))
        for row in histories:
            row["instance"] = instance_id(means)
            row["N_per_arm"] = num_observations_per_arm
        write_csv(result_path, rows)
        write_csv(history_path, histories)
        metadata = {
            "status": "completed",
            "identifier": identifier,
            "completed_at_utc": utc_now(),
            "duration_seconds": time.perf_counter() - start,
            "num_result_rows": len(rows),
            "num_history_rows": len(histories),
            "script_sha256": sha256_file(Path(__file__)),
        }
        atomic_write_json(metadata_path, metadata)
        return metadata
    except Exception as error:  # noqa: BLE001 - toda corrida fallida debe persistirse.
        failure = {
            "status": "failed",
            "identifier": identifier,
            "failed_at_utc": utc_now(),
            "duration_seconds": time.perf_counter() - start,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "script_sha256": sha256_file(Path(__file__)),
        }
        atomic_write_json(metadata_path, failure)
        return failure


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT / "results" / "raw" / "e1")
    args = parser.parse_args()
    rates = load_frozen_rates()
    e0_summary = PROJECT / "results" / "raw" / "e0" / "summary.json"
    if not e0_summary.exists() or '"status": "passed"' not in e0_summary.read_text():
        raise RuntimeError("E0 no está aprobado")
    checkpoint_directory = PROJECT / "checkpoints" / "e1"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger(args.output / "run_ledger.jsonl")
    start = time.perf_counter()
    ledger.append(
        {
            "experiment": "E1",
            "status": "started_or_resumed" if args.resume else "started",
            "rates": rates,
            "workers": args.workers,
        }
    )
    tasks = []
    skipped = []
    for instance_index, means in enumerate(INSTANCES):
        for num_observations_per_arm in E1_SAMPLE_SIZES:
            identifier = f"{instance_id(means)}_N{num_observations_per_arm}"
            metadata_path = checkpoint_directory / f"{identifier}.json"
            result_path = checkpoint_directory / f"{identifier}.csv"
            if args.resume and metadata_path.exists() and result_path.exists():
                metadata_text = metadata_path.read_text(encoding="utf-8")
                if '"status": "completed"' in metadata_text:
                    skipped.append(identifier)
                    continue
            tasks.append(
                (
                    instance_index,
                    means,
                    num_observations_per_arm,
                    rates,
                    str(checkpoint_directory),
                )
            )
    outcomes: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_condition, task) for task in tasks]
        for future in as_completed(futures):
            outcome = future.result()
            outcomes.append(outcome)
            print(
                f"{outcome['identifier']}: {outcome['status']} "
                f"({float(outcome['duration_seconds']):.2f} s)",
                flush=True,
            )
    checkpoint_results = sorted(
        path for path in checkpoint_directory.glob("*.csv") if not path.name.endswith("_history.csv")
    )
    checkpoint_histories = sorted(checkpoint_directory.glob("*_history.csv"))
    all_rows = [row for path in checkpoint_results for row in read_csv_rows(path)]
    all_histories = [row for path in checkpoint_histories for row in read_csv_rows(path)]
    if all_rows:
        write_csv(args.output / "per_arm.csv", all_rows)
    if all_histories:
        write_csv(args.output / "loss_history.csv", all_histories)
    failed = [outcome for outcome in outcomes if outcome["status"] == "failed"]
    summary = {
        "experiment": "E1",
        "status": "completed" if not failed and len(checkpoint_results) == 36 else "incomplete",
        "started_at_or_before_utc": utc_now(),
        "duration_this_invocation_seconds": time.perf_counter() - start,
        "completed_conditions": len(checkpoint_results),
        "expected_conditions": 36,
        "skipped_completed_conditions": skipped,
        "failures": failed,
        "rates": rates,
        "configuration": {
            "num_seeds": len(E1_SEEDS),
            "sample_sizes": E1_SAMPLE_SIZES,
            "num_instances": len(INSTANCES),
            "rho": RHO,
            "steps": NUM_STEPS,
            "batch_size": BATCH_SIZE,
            "initialization": "zeros",
            "workers": args.workers,
        },
        "runtime": runtime_metadata(PROJECT),
        "resume_command": "uv run --locked env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python dev/4_Noprop_fst_Advance/scripts/run_e1.py --workers 4 --resume",
    }
    atomic_write_json(args.output / "summary.json", summary)
    ledger.append(
        {
            "experiment": "E1",
            "status": summary["status"],
            "duration_seconds": summary["duration_this_invocation_seconds"],
            "completed_conditions": summary["completed_conditions"],
            "num_failures": len(failed),
        }
    )
    print(
        f"E1 {summary['status']}: {summary['completed_conditions']}/36 configuraciones "
        f"en {summary['duration_this_invocation_seconds']:.2f} s"
    )
    if summary["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
