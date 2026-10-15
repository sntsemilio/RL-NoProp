#!/usr/bin/env python3
"""Ejecutor reanudable de E2; exige confirmación explícita y no se lanza en este avance."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from noprop_rl.io_utils import RunLedger, atomic_write_json, runtime_metadata, utc_now
from noprop_rl.model import ideal_parameters
from noprop_rl.online import (
    OnlineConfig,
    OnlineRunner,
    RewardSequenceEnvironment,
    evaluate_run,
    generate_reward_sequences,
)
from noprop_rl.protocol import E2_SEEDS, derived_seed, instance_id, load_frozen_rates
from noprop_rl.quadrature import Quadrature


def diagnostic_values(
    runner: OnlineRunner,
    means: tuple[float, float] | None = None,
    include_oracles: bool = False,
) -> dict[str, list[float]]:
    quadrature = Quadrature(80)
    values = []
    for action in range(2):
        if runner.method == "tabular_mean":
            values.append(float(runner._tabular_values()[action]))
        elif runner.method == "d1_mse":
            values.append(quadrature.d1_value(runner.block_1.parameters, action))
        elif runner.method == "d2_mse":
            values.append(quadrature.d2_value(runner.block_2.parameters, action, runner.config.rho))
        else:
            values.append(
                quadrature.composition_value(
                    runner.block_1.parameters, runner.block_2.parameters, action, runner.config.rho
                )
            )
    result = {"learned": values}
    if include_oracles:
        if means is None:
            raise ValueError("Los diagnósticos de oráculo requieren medias solo en el evaluador")
        ideal_1, ideal_2 = ideal_parameters(means, runner.config.rho)
        result["ideal_d1_learned_d2"] = [
            quadrature.composition_value(
                ideal_1, runner.block_2.parameters, action, runner.config.rho
            )
            for action in range(2)
        ]
        result["learned_d1_ideal_d2"] = [
            quadrature.composition_value(
                runner.block_1.parameters, ideal_2, action, runner.config.rho
            )
            for action in range(2)
        ]
    return result


def run_task(arguments: tuple[dict, int, tuple[float, float], str, int, str, str, str]) -> dict:
    config_payload, instance_index, means, method, seed, regime, output_string, task_tag = arguments
    output_directory = Path(output_string)
    identifier = f"{task_tag}_{regime}_{instance_id(means)}_{method}_seed{seed}"
    checkpoint_path = PROJECT / "checkpoints" / "e2" / f"{identifier}.pkl"
    result_path = output_directory / f"{identifier}.json"
    rates = load_frozen_rates()
    if regime == "equal_interactions":
        maximum_interactions = int(config_payload["T"])
        wall_clock_budget = None
    else:
        maximum_interactions = int(config_payload["equal_wall_clock_max_interactions"])
        wall_clock_budget = float(config_payload["equal_wall_clock_budget_seconds"])
    online_config = OnlineConfig(
        horizon=maximum_interactions,
        rho=float(config_payload["rho"]),
        num_inference_replicas=int(config_payload["M"]),
        batch_size=int(config_payload["B"]),
        updates_per_interaction=int(config_payload["U"]),
        learning_rate_mse=rates["mse"],
        learning_rate_bce=rates["bce"],
        balanced_buffer=bool(config_payload.get("balanced_buffer", False)),
        record_event_log=False,
    )
    environment_seed = derived_seed(seed, 1000, instance_index)
    runner_seed = derived_seed(seed, 1001, instance_index)
    previous_elapsed = 0.0
    previous_result = None
    if result_path.exists():
        previous_result = json.loads(result_path.read_text(encoding="utf-8"))
        previous_elapsed = float(previous_result.get("wall_clock_seconds_total", 0.0))
        if wall_clock_budget is not None and previous_elapsed >= wall_clock_budget:
            return previous_result
    if checkpoint_path.exists():
        runner = OnlineRunner.from_checkpoint(checkpoint_path)
    else:
        sequences = generate_reward_sequences(means, maximum_interactions, environment_seed)
        runner = OnlineRunner(RewardSequenceEnvironment(sequences), online_config, method, runner_seed)
    reference_rounds = {int(value) for value in config_payload["reference_rounds"]}
    diagnostics = []
    invocation_start = time.perf_counter()
    while len(runner.actions) < maximum_interactions:
        if (
            wall_clock_budget is not None
            and previous_elapsed + time.perf_counter() - invocation_start >= wall_clock_budget
        ):
            break
        runner.step_once()
        if len(runner.actions) > 2:
            finite_state = bool(
                np.all(np.isfinite(runner.estimated_values[-1]))
                and np.all(np.isfinite(runner.block_1.parameters))
                and np.all(np.isfinite(runner.block_2.parameters))
            )
            if not finite_state:
                runner.save_checkpoint(checkpoint_path)
                failed_result = {
                    "experiment": "E2",
                    "status": "failed_nonfinite",
                    "identifier": identifier,
                    "regime": regime,
                    "instance": instance_id(means),
                    "method": method,
                    "seed": seed,
                    "interactions_completed": len(runner.actions),
                    "checkpoint": str(checkpoint_path),
                }
                atomic_write_json(result_path, failed_result)
                return failed_result
        if len(runner.actions) in reference_rounds:
            values = diagnostic_values(
                runner,
                means,
                include_oracles=bool(config_payload.get("include_oracle_diagnostics", False)),
            )
            diagnostics.append(
                {
                    "round": len(runner.actions),
                    "quadrature_value_arm_1": values["learned"][0],
                    "quadrature_value_arm_2": values["learned"][1],
                    "oracle_diagnostics": {
                        key: value for key, value in values.items() if key != "learned"
                    },
                    "evaluator_partial": evaluate_run(runner.actions, means),
                    "diagnostic_timing_excluded_from_policy_latency": True,
                }
            )
        if len(runner.actions) % 1000 == 0:
            runner.save_checkpoint(checkpoint_path)
    runner.save_checkpoint(checkpoint_path)
    elapsed = time.perf_counter() - invocation_start
    evaluation = evaluate_run(runner.actions, means)
    result = {
        "experiment": "E2",
        "status": "completed",
        "identifier": identifier,
        "regime": regime,
        "instance": instance_id(means),
        "method": method,
        "seed": seed,
        "interactions_completed": len(runner.actions),
        "wall_clock_seconds_this_invocation": elapsed,
        "wall_clock_seconds_total": previous_elapsed + elapsed,
        "wall_clock_budget_seconds": wall_clock_budget,
        "arm_1_visits": runner.actions.count(0),
        "arm_2_visits": runner.actions.count(1),
        "mean_decision_latency_seconds": float(np.mean(runner.decision_durations_seconds)),
        "total_training_time_seconds": float(np.sum(runner.training_durations_seconds)),
        "block_1_updates": runner.block_1.update_count,
        "block_2_updates": runner.block_2.update_count,
        "block_1_evaluations": runner.block_1.evaluation_count,
        "block_2_evaluations": runner.block_2.evaluation_count,
        "diagnostics": diagnostics,
        "evaluator": evaluation,
        "completed_at_utc": utc_now(),
        "checkpoint": str(checkpoint_path),
    }
    atomic_write_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-full", action="store_true")
    parser.add_argument(
        "--regime", choices=("equal_interactions", "equal_wall_clock"), required=True
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "e2_confirmatory.json")
    args = parser.parse_args()
    if not args.confirm_full:
        raise SystemExit(
            "E2 confirmatorio está PENDIENTE y protegido. Añada --confirm-full cuando se autorice."
        )
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    output_directory = PROJECT / "results" / "raw" / "e2_confirmatory" / args.regime
    output_directory.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger(output_directory / "run_ledger.jsonl")
    ledger.append({"experiment": "E2", "status": "started", "regime": args.regime})
    tasks = [
        (
            payload,
            instance_index,
            tuple(means),
            method,
            seed,
            args.regime,
            str(output_directory),
            "e2",
        )
        for instance_index, means in enumerate(payload["instances"])
        for method in payload["methods"]
        for seed in E2_SEEDS
    ]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001 - persistir toda corrida fallida.
                result = {
                    "experiment": "E2",
                    "status": "failed_exception",
                    "instance": instance_id(task[2]),
                    "method": task[3],
                    "seed": task[4],
                    "regime": task[5],
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            results.append(result)
            print(result.get("identifier", result["status"]), flush=True)
    failures = [result for result in results if result["status"] != "completed"]
    summary = {
        "experiment": "E2",
        "status": "completed" if not failures else "incomplete_with_failures",
        "regime": args.regime,
        "num_runs": len(results),
        "failures": failures,
        "runtime": runtime_metadata(PROJECT),
    }
    atomic_write_json(output_directory / "summary.json", summary)
    ledger.append(
        {
            "experiment": "E2",
            "status": summary["status"],
            "regime": args.regime,
            "num_failures": len(failures),
        }
    )


if __name__ == "__main__":
    main()
