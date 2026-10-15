import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural.campaign import jobs
from neural.core import METHODS
from neural.io import ROOT, json_write, now

b = json.loads((ROOT / "results/benchmark_cpu.json").read_text())
rows = {r["method"]: r for r in b["rows"] if r["threads"] == 1}
p = {r["workers"]: r for r in b["parallel"]}
speedup = p[4]["aggregate_updates_per_second"] / p[1]["aggregate_updates_per_second"]
values = {}
for stage in ["N1", "pilot", "N2"]:
    estimates = []
    for j in jobs(stage, {m: 0.01 for m in METHODS}):
        method = j["method"]
        measure = rows.get(
            method,
            rows["structured"] if method == "structured_mse" else rows["logistic"],
        )
        t = measure["train_seconds_5000"]
        q = measure["quadrature_seconds_per_8192_pairs"]
        if stage == "N1":
            q = q * 2 / 8192 * 5
        elif stage == "pilot":
            q = q * 0.5
        else:
            q = q * 1.2  # additional local diagnostics and persistence allowance
        estimates.append(t + q)
    values[stage] = {
        "fits": len(estimates),
        "serial_seconds": sum(estimates),
        "four_worker_estimate_seconds": sum(estimates) / speedup,
        "planning_upper_seconds": sum(estimates) / speedup * 1.5,
    }
result = {
    "utc": now(),
    "measured_parallel_speedup": speedup,
    "stages": values,
    "total_expected_hours": sum(
        x["four_worker_estimate_seconds"] for x in values.values()
    )
    / 3600,
    "total_planning_upper_hours": sum(
        x["planning_upper_seconds"] for x in values.values()
    )
    / 3600,
    "storage_planning_GB": 8,
    "storage_basis": "665? independent fit artifacts; full data in each checkpoint, compressed FP64 predictions, per-run JSON and logs. Conservative 8 GB reserve; actual disk usage monitored.",
    "memory_worker_measured_MiB": p[4]["max_worker_rss_kib"] / 1024,
    "limitations": "Extrapolation from warmed 400-step microbenchmark; ReLU breakpoint count and CPU load change evaluation time; no guarantee.",
}
result["storage_basis"] = (
    "7020 independent fit artifacts, complete observed data and RNG checkpoints, compressed predictions; reserve 8 GB conservatively."
)
json_write(ROOT / "results/resource_estimate.json", result)
print(json.dumps(result, indent=2))
