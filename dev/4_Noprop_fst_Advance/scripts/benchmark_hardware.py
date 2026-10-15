#!/usr/bin/env python3
"""Microbenchmark breve: latencia individual y throughput no se confunden."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from noprop_rl.io_utils import atomic_write_json, runtime_metadata
from noprop_rl.protocol import derived_seed
from noprop_rl.training import fit_blocks_batched

NUM_STEPS = 300
BATCH_SIZE = 64
NUM_REPLICAS = 32


def synthetic_data(num_replicas: int, base_seed: int) -> tuple[np.ndarray, np.ndarray]:
    actions = np.tile(np.repeat(np.arange(2), 512), (num_replicas, 1))
    rewards = []
    for replica in range(num_replicas):
        rng = np.random.default_rng(derived_seed(base_seed, replica))
        rewards.append(np.concatenate((rng.binomial(1, 0.3, 512), rng.binomial(1, 0.7, 512))))
    return actions, np.asarray(rewards, dtype=np.float64)


def cpu_workload(num_replicas: int, base_seed: int, num_steps: int = NUM_STEPS) -> None:
    actions, rewards = synthetic_data(num_replicas, base_seed)
    replica_ids = range(num_replicas)
    fit_blocks_batched(
        actions,
        rewards,
        rho=0.5,
        loss="mse",
        learning_rate=0.01,
        num_steps=num_steps,
        batch_size=BATCH_SIZE,
        seeds_batch=[derived_seed(base_seed, index, 1) for index in replica_ids],
        seeds_noise_1=[derived_seed(base_seed, index, 2) for index in replica_ids],
        seeds_noise_2=[derived_seed(base_seed, index, 3) for index in replica_ids],
        record_every=num_steps,
    )


def process_worker(arguments: tuple[int, int]) -> int:
    num_replicas, base_seed = arguments
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    cpu_workload(num_replicas, base_seed)
    return num_replicas


def measure(function, warmups: int = 1, repetitions: int = 3) -> list[float]:
    for _ in range(warmups):
        function()
    durations = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        durations.append(time.perf_counter() - start)
    return durations


def gpu_workload(num_replicas: int, base_seed: int, num_steps: int = NUM_STEPS) -> None:
    import torch

    device = torch.device("cuda")
    actions_numpy, rewards_numpy = synthetic_data(num_replicas, base_seed)
    plans = []
    noise_1 = []
    noise_2 = []
    for replica in range(num_replicas):
        plans.append(
            np.random.default_rng(derived_seed(base_seed, replica, 1)).integers(
                0, actions_numpy.shape[1], size=(num_steps, BATCH_SIZE)
            )
        )
        noise_1.append(
            np.random.default_rng(derived_seed(base_seed, replica, 2)).standard_normal(
                (num_steps, BATCH_SIZE)
            )
        )
        noise_2.append(
            np.random.default_rng(derived_seed(base_seed, replica, 3)).standard_normal(
                (num_steps, BATCH_SIZE)
            )
        )
    actions = torch.as_tensor(actions_numpy, dtype=torch.long, device=device)
    rewards = torch.as_tensor(rewards_numpy, dtype=torch.float64, device=device)
    indices = torch.as_tensor(np.stack(plans), dtype=torch.long, device=device)
    noises = [
        torch.as_tensor(np.stack(noise_1), dtype=torch.float64, device=device),
        torch.as_tensor(np.stack(noise_2), dtype=torch.float64, device=device),
    ]
    rows = torch.arange(num_replicas, device=device)[:, None]
    for block in range(2):
        theta = torch.zeros((num_replicas, 3), dtype=torch.float64, device=device)
        for step in range(num_steps):
            action_batch = actions[rows, indices[:, step, :]]
            reward_batch = rewards[rows, indices[:, step, :]]
            z = noises[block][:, step, :]
            if block == 1:
                z = np.sqrt(0.5) * reward_batch + np.sqrt(0.5) * z
            logits = theta[:, 0, None] * z + torch.gather(theta[:, 1:], 1, action_batch)
            probabilities = torch.sigmoid(logits)
            factors = (probabilities - reward_batch) * probabilities * (1.0 - probabilities)
            gradient = torch.stack(
                (
                    torch.mean(factors * z, dim=1),
                    torch.mean(factors * (action_batch == 0), dim=1),
                    torch.mean(factors * (action_batch == 1), dim=1),
                ),
                dim=1,
            )
            theta -= 0.01 * gradient
    torch.cuda.synchronize()


def main() -> None:
    output = PROJECT / "results" / "hardware_benchmark.json"
    vectorized = measure(lambda: cpu_workload(NUM_REPLICAS, 81000))
    individual = measure(lambda: cpu_workload(1, 82000))

    def multiprocessing_run() -> None:
        with ProcessPoolExecutor(max_workers=4) as pool:
            list(pool.map(process_worker, [(8, 83000 + index) for index in range(4)]))

    processes = measure(multiprocessing_run)
    payload: dict[str, object] = {
        "workload": {
            "steps": NUM_STEPS,
            "batch_size": BATCH_SIZE,
            "blocks": 2,
            "batched_replicas": NUM_REPLICAS,
            "dtype": "float64",
            "warmups": 1,
            "measured_repetitions": 3,
        },
        "cpu_vectorized": {
            "durations_seconds": vectorized,
            "median_seconds": float(np.median(vectorized)),
            "throughput_replicas_per_second": NUM_REPLICAS / float(np.median(vectorized)),
        },
        "cpu_processes_4": {
            "durations_seconds": processes,
            "median_seconds": float(np.median(processes)),
            "throughput_replicas_per_second": NUM_REPLICAS / float(np.median(processes)),
        },
        "cpu_individual_latency": {
            "durations_seconds": individual,
            "median_seconds": float(np.median(individual)),
        },
        "runtime": runtime_metadata(PROJECT),
        "host": {
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "load_average": os.getloadavg(),
        },
    }
    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            gpu_batch = measure(lambda: gpu_workload(NUM_REPLICAS, 84000))
            gpu_individual = measure(lambda: gpu_workload(1, 85000))
            free_memory, total_memory = torch.cuda.mem_get_info()
            payload["gpu"] = {
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "free_memory_after_benchmark_bytes": free_memory,
                "reported_total_memory_after_benchmark_bytes": total_memory,
                "backend": "CUDA",
                "torch_cuda_version": torch.version.cuda,
                "batched_durations_seconds": gpu_batch,
                "batched_median_seconds": float(np.median(gpu_batch)),
                "throughput_replicas_per_second": NUM_REPLICAS / float(np.median(gpu_batch)),
                "individual_durations_seconds": gpu_individual,
                "individual_median_seconds": float(np.median(gpu_individual)),
                "synchronized": True,
                "nvidia_smi_note": "NVML blocked by operating system in this managed environment",
            }
        else:
            payload["gpu"] = {"available": False}
    except (RuntimeError, OSError) as error:  # La falla se registra y no impide comparar CPU.
        payload["gpu"] = {"available": False, "error": repr(error)}
    candidates = {
        "cpu_vectorized": payload["cpu_vectorized"]["throughput_replicas_per_second"],
        "cpu_processes_4": payload["cpu_processes_4"]["throughput_replicas_per_second"],
    }
    if isinstance(payload.get("gpu"), dict) and "throughput_replicas_per_second" in payload["gpu"]:
        candidates["gpu_batched"] = payload["gpu"]["throughput_replicas_per_second"]
    payload["decision"] = {
        "selected_for_e1": max(candidates, key=candidates.get),
        "basis": "highest measured batched throughput for equivalent updates",
        "individual_latency_reported_separately": True,
    }
    atomic_write_json(output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
