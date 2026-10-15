import argparse
import concurrent.futures
import multiprocessing
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from extension import ROOT
from extension.agents import InteractiveAgent
from extension.environment import PrivateEnvironment
from extension.io import config, json_write, now
from neural.quadrature import readouts


def measure(method, device="cpu", steps=400, quadrature=True):
    torch.set_num_threads(1)
    agent = InteractiveAgent(method, 90020, 0.1, capacity=20000, device=device)
    env = PrivateEnvironment(90020, "sine", 2)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def loop(n):
        totals = {"decision": 0.0, "update": 0.0, "environment": 0.0}
        for _ in range(n):
            start = time.perf_counter()
            c = env.context()
            totals["environment"] += time.perf_counter() - start
            sync()
            start = time.perf_counter()
            a = agent.choose(c)
            sync()
            totals["decision"] += time.perf_counter() - start
            start = time.perf_counter()
            r = env.step(a)
            totals["environment"] += time.perf_counter() - start
            sync()
            start = time.perf_counter()
            agent.observe(c, a, r)
            sync()
            totals["update"] += time.perf_counter() - start
        return totals

    warm = loop(50)
    early = loop(steps)
    # Fill with actual observed simulator triples, solely for memory/growth timing.
    start = time.perf_counter()
    while agent.size < 18000:
        c = env.context()
        a = agent.size % 2
        r = env.step(a)
        agent.c[agent.size] = torch.as_tensor(c, device=device)
        agent.a[agent.size] = a
        agent.r[agent.size] = r
        agent.size += 1
    sync()
    fill = time.perf_counter() - start
    late = loop(steps)
    quad = None
    if quadrature:
        c = np.random.default_rng(90021).uniform(-1, 1, (128, 2))
        a = np.arange(128) % 2
        start = time.perf_counter()
        readouts(agent.learner, c, a)
        quad = time.perf_counter() - start
    return {
        "method": method,
        "device": device,
        "warmup_seconds": sum(warm.values()),
        "early": early,
        "late": late,
        "steps_per_phase": steps,
        "seconds_per_interaction": (sum(early.values()) + sum(late.values()))
        / (2 * steps),
        "buffer_fill_seconds": fill,
        "buffer_capacity": 20000,
        "final_buffer_size": agent.size,
        "quadrature_seconds_per128pairs": quad,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def concurrent_probe(_):
    return measure("structured", steps=600, quadrature=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this process")
    inventory = {
        "utc": now(),
        "cpus": os.cpu_count(),
        "affinity": len(os.sched_getaffinity(0)),
        "load": os.getloadavg(),
        "memory": Path("/proc/meminfo").read_text(),
        "disk": shutil.disk_usage(ROOT)._asdict(),
        "cpu": subprocess.check_output(["lscpu"], text=True),
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    rows = []
    for m in config("base")["methods"]:
        row = measure(m, args.device)
        rows.append(row)
        print(m, row["seconds_per_interaction"], flush=True)
    concurrency = []
    if args.device == "cpu":
        for workers in (2, 4):
            start = time.perf_counter()
            with concurrent.futures.ProcessPoolExecutor(
                workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                rr = list(pool.map(concurrent_probe, range(workers)))
            concurrency.append(
                {
                    "workers": workers,
                    "wall_with_startup": time.perf_counter() - start,
                    "aggregate_interactions_per_second": sum(
                        1 / r["seconds_per_interaction"] for r in rr
                    ),
                    "peak_worker_rss_kib": max(r["peak_rss_kib"] for r in rr),
                }
            )
    output = {"inventory": inventory, "rows": rows, "concurrency": concurrency}
    json_write(ROOT / f"results/benchmark_{args.device}.json", output)
    if args.device == "cpu":
        mean = np.mean([r["seconds_per_interaction"] for r in rows])
        pilot = 1920 * 5000 * mean
        interactions = 720 * 20000 * mean
        direct = next(
            r["seconds_per_interaction"] for r in rows if r["method"] == "direct"
        )
        timed = 720 * 20000 * direct
        # Each main fixed-T run has 4 diagnostics; time runs at most 5 including final.
        diagnostics = (
            720 * 9 * np.mean([r["quadrature_seconds_per128pairs"] * 64 for r in rows])
        )
        estimate = {
            "utc": now(),
            "workers": 4,
            "offline_N2A_fits": 480,
            "pilot_fits": 1920,
            "N3_fixed_fits": 720,
            "N3_time_fits": 720,
            "serial_seconds": {
                "N2A": 480 * 8,
                "pilot": pilot,
                "N3_interactions": interactions,
                "N3_time": timed,
                "diagnostics": diagnostics,
            },
            "estimated_wall_hours": (
                480 * 8 + pilot + interactions + timed + diagnostics
            )
            / 4
            / 3600,
            "estimated_storage_gib": 8,
            "storage_budget_note": "conservative 8 GiB including checkpoints, trajectories, predictions and source snapshots",
            "timing_uncertainty": "initially estimated; training load and trained ReLU quadrature can change; no sample/step reduction",
            "policy": "CPU selected only after GPU comparison; four single-thread workers if memory and throughput permit",
        }
        json_write(ROOT / "results/resource_estimate.json", estimate)
        print(estimate, flush=True)
