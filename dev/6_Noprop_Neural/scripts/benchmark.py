import argparse
import concurrent.futures
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from neural.core import Learner, generator, sample
from neural.environment import triples
from neural.io import ROOT, json_write, now, provenance
from neural.quadrature import readouts


def trial(device="cpu", threads=1, steps=400):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    c, a, r = triples(81100, "interaction", 2, 2048, "benchmark")
    c, a, r = [torch.as_tensor(x, device=device) for x in (c, a, r)]
    rows = []

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    for method in ["direct", "structured", "flexible", "ensemble", "wide", "logistic"]:
        learner = Learner(method, 81100, 0.01, device=device)
        warm = time.perf_counter()
        for _ in range(50):
            learner.update(c, a, r)
        sync()
        warm = time.perf_counter() - warm
        sync()
        t = time.perf_counter()
        for _ in range(steps):
            learner.update(c, a, r)
        sync()
        seconds = time.perf_counter() - t
        infer = []
        for _ in range(3):
            sync()
            t = time.perf_counter()
            if learner.composed:
                sample(
                    learner,
                    c[:2],
                    a[:2],
                    32,
                    [generator(81100, "infer", j, device=device) for j in range(3)],
                )
            else:
                with torch.no_grad():
                    for b in learner.blocks:
                        b(c[:2], a[:2], r[:2]).sigmoid()
            sync()
            infer.append(time.perf_counter() - t)
        t = time.perf_counter()
        readouts(learner, c[:64].cpu().numpy(), a[:64].cpu().numpy())
        q = time.perf_counter() - t
        rows.append(
            {
                "method": method,
                "device": device,
                "threads": threads,
                "warmup_seconds": warm,
                "updates_per_second": steps / seconds,
                "train_seconds_5000": seconds / steps * 5000,
                "inference_seconds_two_pairs_M32": float(np.median(infer[1:])),
                "quadrature_seconds_per_8192_pairs": q / 64 * 8192,
            }
        )
    return rows


def parallel_worker(_):
    torch.set_num_threads(1)
    c, a, r = map(
        torch.from_numpy, triples(81101, "linear", 2, 2048, "benchmark_parallel")
    )
    l = Learner("structured", 81101, 0.01)
    for _ in range(50):
        l.update(c, a, r)
    t = time.perf_counter()
    for _ in range(1000):
        l.update(c, a, r)
    return time.perf_counter() - t, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()
    inventory = {
        "utc": now(),
        "cpu": platform.processor(),
        "cpus": os.cpu_count(),
        "affinity": len(os.sched_getaffinity(0)),
        "load": os.getloadavg(),
        "meminfo": Path("/proc/meminfo").read_text(),
        "provenance": provenance(),
        "cuda_available": torch.cuda.is_available(),
    }
    for command in [["lscpu"], ["df", "-h", str(ROOT)], ["nvidia-smi"]]:
        s = subprocess.run(command, capture_output=True, text=True, check=False)
        inventory[" ".join(command)] = s.stdout + s.stderr
    if args.device == "cuda" and not torch.cuda.is_available():
        json_write(
            ROOT / "results/benchmark_cuda.json",
            {"inventory": inventory, "status": "unavailable"},
        )
        sys.exit(0)
    rows = []
    for threads in [1, 2, 4] if args.device == "cpu" else [1]:
        rows += trial(args.device, threads)
    parallel = []
    if args.device == "cpu":
        import multiprocessing as mp

        for workers in [1, 2, 4]:
            with concurrent.futures.ProcessPoolExecutor(
                workers, mp_context=mp.get_context("spawn")
            ) as pool:
                output = list(pool.map(parallel_worker, range(workers)))
            parallel.append(
                {
                    "workers": workers,
                    "aggregate_updates_per_second": 1000
                    * workers
                    / max(v[0] for v in output),
                    "max_worker_rss_kib": max(v[1] for v in output),
                }
            )
    result = {"inventory": inventory, "rows": rows, "parallel": parallel}
    json_write(ROOT / f"results/benchmark_{args.device}.json", result)
    print(json.dumps({"rows": rows, "parallel": parallel}, indent=2))
