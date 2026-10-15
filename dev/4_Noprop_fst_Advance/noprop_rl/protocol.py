"""Constantes del protocolo, sin acceso implícito a medias durante el aprendizaje."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

INSTANCES = [
    (0.30, 0.70),
    (0.70, 0.30),
    (0.45, 0.55),
    (0.55, 0.45),
    (0.49, 0.51),
    (0.51, 0.49),
    (0.05, 0.95),
    (0.95, 0.05),
    (0.50, 0.50),
]
PILOT_INSTANCES = [(0.20, 0.80), (0.40, 0.60)]
E0_RHOS = [0.1, 0.5, 0.9]
E1_SAMPLE_SIZES = [32, 128, 512, 2048]
LEARNING_RATE_CANDIDATES = [1e-3, 1e-2, 1e-1]
PILOT_SEEDS = list(range(21000, 21010))
E1_SEEDS = list(range(31000, 31030))
E2_SEEDS = list(range(41000, 41050))
E3_SEEDS = list(range(51000, 51030))
SMOKE_SEEDS = [61000, 61001]
BOOTSTRAP_SEED = 71000


def derived_seed(base_seed: int, *labels: int) -> int:
    payload = json.dumps([base_seed, *labels], separators=(",", ":")).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def instance_id(means: tuple[float, float]) -> str:
    return f"mu1_{means[0]:.2f}_mu2_{means[1]:.2f}"


def project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def load_frozen_rates() -> dict[str, float]:
    path = project_dir() / "configs" / "frozen_rates.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen_before_e1":
        raise RuntimeError("Las tasas no están congeladas para E1")
    return {"mse": float(payload["rates"]["mse"]), "bce": float(payload["rates"]["bce"])}
