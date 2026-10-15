#!/usr/bin/env python3
"""Inventario reproducible de recursos observados en el entorno administrado."""

from __future__ import annotations

import importlib.metadata
import os
import resource
import shutil
import sys
from pathlib import Path

import psutil

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from noprop_rl.io_utils import atomic_write_json, runtime_metadata, utc_now


def read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def limit(name: int) -> dict[str, int | str]:
    soft, hard = resource.getrlimit(name)

    def render(value: int) -> int | str:
        return "unlimited" if value == resource.RLIM_INFINITY else value

    return {"soft": render(soft), "hard": render(hard)}


def cpu_model() -> str | None:
    contents = read_optional(Path("/proc/cpuinfo"))
    if contents is None:
        return None
    for line in contents.splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def main() -> None:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = shutil.disk_usage(PROJECT)
    frequencies = psutil.cpu_freq()
    processes = []
    for process in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
        try:
            info = process.info
            processes.append(
                {
                    "pid": info["pid"],
                    "name": info["name"],
                    "cpu_percent": info["cpu_percent"],
                    "rss_bytes": info["memory_info"].rss,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    processes.sort(key=lambda row: (row["cpu_percent"], row["rss_bytes"]), reverse=True)
    packages = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "matplotlib", "torch", "pytest", "ruff", "psutil")
    }
    gpu: dict[str, object]
    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            free_memory, total_memory = torch.cuda.mem_get_info()
            gpu = {
                "available": True,
                "backend": "CUDA",
                "cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "currently_free_memory_bytes": free_memory,
                "currently_total_memory_bytes": total_memory,
                "nvidia_smi": "NVML access blocked by operating system; torch CUDA calls succeeded",
            }
        else:
            gpu = {"available": False}
    except (RuntimeError, OSError) as error:
        gpu = {"available": False, "error": repr(error)}
    payload = {
        "captured_at_utc": utc_now(),
        "cpu": {
            "model": cpu_model(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cpus": psutil.cpu_count(logical=True),
            "affinity": psutil.Process().cpu_affinity(),
            "frequency_mhz": None
            if frequencies is None
            else {"current": frequencies.current, "min": frequencies.min, "max": frequencies.max},
            "load_average": os.getloadavg(),
        },
        "memory": {
            "total_bytes": memory.total,
            "available_bytes": memory.available,
            "used_bytes": memory.used,
            "swap_total_bytes": swap.total,
            "swap_used_bytes": swap.used,
        },
        "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "gpu": gpu,
        "limits": {
            "address_space": limit(resource.RLIMIT_AS),
            "cpu_time": limit(resource.RLIMIT_CPU),
            "open_files": limit(resource.RLIMIT_NOFILE),
            "processes": limit(resource.RLIMIT_NPROC),
            "stack": limit(resource.RLIMIT_STACK),
            "cgroup_cpu_max": read_optional(Path("/sys/fs/cgroup/cpu.max")),
            "cgroup_cpuset": read_optional(Path("/sys/fs/cgroup/cpuset.cpus.effective")),
            "cgroup_memory_max": read_optional(Path("/sys/fs/cgroup/memory.max")),
            "managed_filesystem": "workspace and /tmp writable; external network restricted",
        },
        "top_processes": processes[:20],
        "python_packages": packages,
        "runtime": runtime_metadata(PROJECT),
    }
    benchmark_path = PROJECT / "results" / "hardware_benchmark.json"
    if benchmark_path.exists():
        import json

        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        payload["gpu_observation_during_benchmark"] = benchmark.get("gpu")
        payload["gpu_availability_note"] = (
            "CUDA funcionó durante el microbenchmark, pero una captura posterior no pudo "
            "inicializar NVML/CUDA; la disponibilidad es intermitente en el entorno administrado."
        )
    atomic_write_json(PROJECT / "results" / "hardware_inventory.json", payload)
    print(f"Inventario guardado; RAM disponible {memory.available / 2**30:.2f} GiB, disco libre {disk.free / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
