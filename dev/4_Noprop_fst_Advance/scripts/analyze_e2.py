#!/usr/bin/env python3
"""Agregación futura de E2: ocho instancias normalizadas y empate separado."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from noprop_rl.io_utils import atomic_write_json, write_csv
from noprop_rl.metrics import bootstrap_interval
from noprop_rl.protocol import BOOTSTRAP_SEED, derived_seed

NUM_BOOTSTRAP = 10_000


def summary(values: list[float], seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean_low, mean_high = bootstrap_interval(
        array, np.random.default_rng(derived_seed(seed, 1)), NUM_BOOTSTRAP, "mean"
    )
    median_low, median_high = bootstrap_interval(
        array, np.random.default_rng(derived_seed(seed, 2)), NUM_BOOTSTRAP, "median"
    )
    return {
        "n": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "mean_ci95_low": mean_low,
        "mean_ci95_high": mean_high,
        "median_ci95_low": median_low,
        "median_ci95_high": median_high,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=PROJECT / "results" / "raw" / "e2_confirmatory"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "results" / "aggregated" / "e2_confirmatory"
    )
    args = parser.parse_args()
    result_files = [path for path in args.input.rglob("*.json") if path.name != "summary.json"]
    if not result_files:
        raise SystemExit("E2 sigue pendiente: no hay resultados confirmatorios para agregar.")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_files]
    failures = [row for row in results if row.get("status") != "completed"]
    completed = [row for row in results if row.get("status") == "completed"]
    non_tied: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    tied_rows = []
    per_instance_rows = []
    point_groups: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in completed:
        evaluator = row["evaluator"]
        regime = row["regime"]
        if float(evaluator["gap"]) == 0.0:
            tied_rows.append(
                {
                    "regime": regime,
                    "instance": row["instance"],
                    "method": row["method"],
                    "seed": row["seed"],
                    "arm_1_fraction": row["arm_1_visits"] / row["interactions_completed"],
                    "pseudoregret": evaluator["pseudoregret"],
                }
            )
            continue
        normalized = float(evaluator["normalized_pseudoregret"])
        non_tied[(regime, row["method"], int(row["seed"]))].append(normalized)
        per_instance_rows.append(
            {
                "regime": regime,
                "instance": row["instance"],
                "method": row["method"],
                "seed": row["seed"],
                "normalized_pseudoregret": normalized,
                "interactions_completed": row["interactions_completed"],
            }
        )
        for diagnostic in row.get("diagnostics", []):
            partial = diagnostic["evaluator_partial"]
            point_groups[(regime, row["method"], int(diagnostic["round"]))].append(
                float(partial["normalized_pseudoregret"])
            )
    seed_aggregates = {
        key: float(np.mean(values)) for key, values in non_tied.items() if len(values) == 8
    }
    method_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (regime, method, _seed), value in seed_aggregates.items():
        method_groups[(regime, method)].append(value)
    aggregate_rows = []
    for index, ((regime, method), values) in enumerate(sorted(method_groups.items())):
        aggregate_rows.append(
            {
                "regime": regime,
                "method": method,
                "aggregation": "mean RT/(T*gap) across eight non-tied instances per complete seed",
                **summary(values, derived_seed(BOOTSTRAP_SEED, 80, index)),
                "interval_type": "95% percentile bootstrap over complete seeds",
                "bootstrap_resamples": NUM_BOOTSTRAP,
            }
        )
    paired_rows = []
    regimes = sorted({key[0] for key in seed_aggregates})
    methods = sorted({key[1] for key in seed_aggregates if key[1] != "tabular_mean"})
    for regime_index, regime in enumerate(regimes):
        for method_index, method in enumerate(methods):
            differences = []
            for seed in sorted({key[2] for key in seed_aggregates if key[0] == regime}):
                method_key = (regime, method, seed)
                control_key = (regime, "tabular_mean", seed)
                if method_key in seed_aggregates and control_key in seed_aggregates:
                    differences.append(seed_aggregates[method_key] - seed_aggregates[control_key])
            if differences:
                paired_rows.append(
                    {
                        "regime": regime,
                        "method": method,
                        "control": "tabular_mean",
                        "difference": "method minus control; negative favors method",
                        **summary(
                            differences,
                            derived_seed(BOOTSTRAP_SEED, 81, regime_index, method_index),
                        ),
                        "paired": True,
                    }
                )
    point_rows = []
    for index, ((regime, method, round_number), values) in enumerate(sorted(point_groups.items())):
        point_rows.append(
            {
                "regime": regime,
                "method": method,
                "round": round_number,
                **summary(values, derived_seed(BOOTSTRAP_SEED, 82, index)),
                "interval_type": "pointwise 95% bootstrap; not simultaneous",
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "per_instance_seed.csv", per_instance_rows)
    write_csv(args.output / "normalized_eight_instances.csv", aggregate_rows)
    write_csv(args.output / "paired_vs_tabular.csv", paired_rows)
    write_csv(args.output / "tie_separate.csv", tied_rows)
    write_csv(args.output / "pointwise_curves.csv", point_rows)
    atomic_write_json(
        args.output / "summary.json",
        {
            "status": "completed",
            "num_input_runs": len(results),
            "num_failed_runs": len(failures),
            "failures": failures,
            "tie_divided_by_zero": False,
            "bootstrap_resamples": NUM_BOOTSTRAP,
            "pointwise_intervals_are_not_simultaneous": True,
        },
    )


if __name__ == "__main__":
    main()
