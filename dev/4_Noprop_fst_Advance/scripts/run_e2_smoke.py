#!/usr/bin/env python3
"""Smoke test no confirmatorio del flujo online y su reanudación."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from noprop_rl.io_utils import RunLedger, atomic_write_json, runtime_metadata, write_csv
from noprop_rl.online import (
    OnlineConfig,
    OnlineRunner,
    RewardSequenceEnvironment,
    evaluate_run,
    generate_reward_sequences,
)
from noprop_rl.protocol import SMOKE_SEEDS, derived_seed, instance_id, load_frozen_rates

SMOKE_INSTANCES = [(0.30, 0.70), (0.50, 0.50)]
SMOKE_HORIZON = 128
SMOKE_M = 8
METHODS = ["two_block_mse", "tabular_mean"]


def event_order_is_valid(runner: OnlineRunner) -> bool:
    if len(runner.event_log) != 4 * runner.config.horizon:
        return False
    expected = ["decision", "observation", "buffer_append", "update"]
    for round_index in range(runner.config.horizon):
        events = runner.event_log[4 * round_index : 4 * (round_index + 1)]
        if [event["event"] for event in events] != expected:
            return False
        if any(int(event["round"]) != round_index + 1 for event in events):
            return False
    return True


def runners_equivalent(first: OnlineRunner, second: OnlineRunner) -> bool:
    return bool(
        first.actions == second.actions
        and first.rewards == second.rewards
        and first.buffer_actions == second.buffer_actions
        and first.buffer_rewards == second.buffer_rewards
        and np.array_equal(first.block_1.parameters, second.block_1.parameters)
        and np.array_equal(first.block_2.parameters, second.block_2.parameters)
        and np.array_equal(first.environment.positions, second.environment.positions)
        and np.allclose(first.estimated_values, second.estimated_values, equal_nan=True)
        and first.losses == second.losses
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT / "results" / "raw" / "e2_smoke")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = PROJECT / "checkpoints" / "e2_smoke"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    rates = load_frozen_rates()
    config = OnlineConfig(
        horizon=SMOKE_HORIZON,
        rho=0.5,
        num_inference_replicas=SMOKE_M,
        batch_size=64,
        updates_per_interaction=1,
        learning_rate_mse=rates["mse"],
        learning_rate_bce=rates["bce"],
    )
    ledger = RunLedger(args.output / "run_ledger.jsonl")
    ledger.append({"experiment": "E2_smoke", "status": "started", "non_confirmatory": True})
    start = time.perf_counter()
    summaries: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []
    all_passed = True
    for instance_index, means in enumerate(SMOKE_INSTANCES):
        for method_index, method in enumerate(METHODS):
            for seed in SMOKE_SEEDS:
                environment_seed = derived_seed(seed, 900, instance_index)
                reward_sequences = generate_reward_sequences(means, SMOKE_HORIZON, environment_seed)
                runner_seed = derived_seed(seed, 901, method_index)
                full = OnlineRunner(
                    RewardSequenceEnvironment(reward_sequences), config, method, runner_seed
                )
                full.run()
                checkpoint_path = checkpoint_directory / (
                    f"{instance_id(means)}_{method}_seed{seed}.pkl"
                )
                partial = OnlineRunner(
                    RewardSequenceEnvironment(reward_sequences), config, method, runner_seed
                )
                partial.run(until=63)
                partial.save_checkpoint(checkpoint_path)
                resumed = OnlineRunner.from_checkpoint(checkpoint_path)
                resumed.run()
                resumed.save_checkpoint(
                    checkpoint_directory / f"{instance_id(means)}_{method}_seed{seed}_complete.pkl"
                )

                evaluation = evaluate_run(full.actions, means)
                resume_exact = runners_equivalent(full, resumed)
                order_valid = event_order_is_valid(full)
                counts_valid = bool(
                    len(full.actions)
                    == len(full.rewards)
                    == len(full.buffer_actions)
                    == len(full.buffer_rewards)
                    == SMOKE_HORIZON
                    and int(np.sum(full.environment.positions)) == SMOKE_HORIZON
                )
                initialization_valid = full.actions[:2] == [0, 1]
                real_buffer_only = full.synthetic_buffer_insertions == 0 and all(
                    event.get("real_environment_reward", True)
                    for event in full.event_log
                    if event["event"] == "buffer_append"
                )
                information_isolated = bool(
                    not hasattr(full, "means")
                    and not hasattr(full, "gap")
                    and not hasattr(full.environment, "means")
                )
                streams = {
                    name: derived_seed(runner_seed, label)
                    for name, label in OnlineRunner._STREAM_LABELS.items()
                }
                streams_distinct = len(set(streams.values())) == len(streams)
                run_passed = all(
                    (
                        resume_exact,
                        order_valid,
                        counts_valid,
                        initialization_valid,
                        real_buffer_only,
                        information_isolated,
                        streams_distinct,
                    )
                )
                all_passed = all_passed and run_passed
                summaries.append(
                    {
                        "instance": instance_id(means),
                        "method": method,
                        "seed": seed,
                        "status": "passed" if run_passed else "failed",
                        "non_confirmatory": True,
                        "horizon": SMOKE_HORIZON,
                        "M": SMOKE_M,
                        "resume_bitwise_exact": resume_exact,
                        "event_order_valid": order_valid,
                        "counts_valid": counts_valid,
                        "forced_initialization_valid": initialization_valid,
                        "real_buffer_only": real_buffer_only,
                        "information_isolated": information_isolated,
                        "rng_streams_distinct": streams_distinct,
                        "rng_stream_seeds": streams,
                        "arm_1_visits": full.actions.count(0),
                        "arm_2_visits": full.actions.count(1),
                        "block_1_updates": full.block_1.update_count,
                        "block_2_updates": full.block_2.update_count,
                        "block_1_evaluations": full.block_1.evaluation_count,
                        "block_2_evaluations": full.block_2.evaluation_count,
                        "decision_latency_mean_seconds": float(
                            np.mean(full.decision_durations_seconds)
                        ),
                        "decision_latency_median_seconds": float(
                            np.median(full.decision_durations_seconds)
                        ),
                        "training_time_total_seconds": float(
                            np.sum(full.training_durations_seconds)
                        ),
                        **evaluation,
                    }
                )
                cumulative_regret = 0.0
                optimal_mean = max(means)
                for round_index, (action, reward, values) in enumerate(
                    zip(full.actions, full.rewards, full.estimated_values, strict=True), start=1
                ):
                    cumulative_regret += optimal_mean - means[action]
                    round_rows.append(
                        {
                            "instance": instance_id(means),
                            "method": method,
                            "seed": seed,
                            "round": round_index,
                            "action": action + 1,
                            "reward": reward,
                            "estimated_value_arm_1": values[0],
                            "estimated_value_arm_2": values[1],
                            "cumulative_pseudoregret_evaluator_only": cumulative_regret,
                            "buffer_size": round_index,
                        }
                    )
    duration = time.perf_counter() - start
    write_csv(args.output / "per_run.csv", summaries)
    write_csv(args.output / "per_round.csv", round_rows)
    summary = {
        "experiment": "E2_smoke",
        "status": "passed" if all_passed else "failed",
        "non_confirmatory": True,
        "scope": {
            "instances": SMOKE_INSTANCES,
            "seeds": SMOKE_SEEDS,
            "methods": METHODS,
            "horizon": SMOKE_HORIZON,
            "M": SMOKE_M,
            "confirmatory_e2_not_executed": True,
        },
        "duration_seconds": duration,
        "num_runs": len(summaries),
        "runtime": runtime_metadata(PROJECT),
    }
    atomic_write_json(args.output / "summary.json", summary)
    ledger.append(
        {
            "experiment": "E2_smoke",
            "status": summary["status"],
            "duration_seconds": duration,
            "non_confirmatory": True,
        }
    )
    print(f"E2 smoke {summary['status']}: {len(summaries)} corridas en {duration:.2f} s")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
