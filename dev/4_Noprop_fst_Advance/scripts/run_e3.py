#!/usr/bin/env python3
"""Materializa E3 OFAT; protegido contra ejecución accidental."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from noprop_rl.io_utils import atomic_write_json, runtime_metadata
from noprop_rl.protocol import E3_SEEDS
from run_e2 import run_task


def build_manifest(config: dict) -> list[dict[str, object]]:
    base = config["base"]
    rows = []
    row_by_key = {}
    for factor, values in config["one_factor_at_a_time_sweeps"].items():
        for value in values:
            setting = dict(base)
            setting[factor] = value
            key = tuple(sorted(setting.items()))
            membership = f"{factor}={value}"
            if key in row_by_key:
                row_by_key[key]["sweep_membership"].append(membership)
                continue
            row = {
                "factor": factor,
                "value": value,
                "sweep_membership": [membership],
                "setting": setting,
                "instances": config["instances"],
                "num_seeds": config["num_seeds_per_configuration"],
            }
            rows.append(row)
            row_by_key[key] = row
    for ablation in config["additional_ablations"]:
        rows.append(
            {
                "factor": "ablation",
                "value": ablation,
                "sweep_membership": [f"ablation={ablation}"],
                "setting": dict(base),
                "instances": config["instances"],
                "num_seeds": config["num_seeds_per_configuration"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-full", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--index", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "e3.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = build_manifest(config)
    output = PROJECT / "configs" / "e3_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifiesto E3: {len(manifest)} configuraciones en {output}")
    if args.manifest_only:
        return
    if not args.confirm_full:
        raise SystemExit("E3 está PENDIENTE. Use --manifest-only o autorice con --confirm-full.")
    if args.index is None or not 0 <= args.index < len(manifest):
        raise SystemExit(f"Seleccione --index entre 0 y {len(manifest) - 1}; se ejecuta un factor a la vez.")
    selected = manifest[args.index]
    setting = selected["setting"]
    ablation = selected["value"] if selected["factor"] == "ablation" else None
    method = "mean_substitute_mse" if ablation == "mean_substitute" else "two_block_mse"
    payload = {
        "T": setting["T"],
        "rho": setting["rho"],
        "M": setting["M"],
        "B": setting["B"],
        "U": setting["U"],
        "reference_rounds": [10, 100, 1000, 10000],
        "balanced_buffer": ablation == "balanced_buffer_by_arm",
        "include_oracle_diagnostics": True,
        "equal_wall_clock_max_interactions": 1000000,
        "equal_wall_clock_budget_seconds": 10.0,
    }
    output_directory = PROJECT / "results" / "raw" / "e3" / f"configuration_{args.index}"
    output_directory.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            payload,
            instance_index,
            tuple(means),
            method,
            seed,
            "equal_interactions",
            str(output_directory),
            f"e3cfg{args.index}",
        )
        for instance_index, means in enumerate(config["instances"])
        for seed in E3_SEEDS
    ]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(result["identifier"], flush=True)
    atomic_write_json(
        output_directory / "summary.json",
        {
            "experiment": "E3",
            "status": "completed"
            if all(result["status"] == "completed" for result in results)
            else "incomplete_with_failures",
            "configuration_index": args.index,
            "configuration": selected,
            "num_runs": len(results),
            "failures": [result for result in results if result["status"] != "completed"],
            "runtime": runtime_metadata(PROJECT),
        },
    )


if __name__ == "__main__":
    main()
