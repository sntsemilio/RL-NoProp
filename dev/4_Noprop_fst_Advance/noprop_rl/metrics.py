"""Métricas por brazo y bootstrap con la semilla como unidad experimental."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    predictions: np.ndarray, labels: np.ndarray, num_bins: int = 10
) -> float:
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_ids = np.minimum(np.searchsorted(edges, predictions, side="right") - 1, num_bins - 1)
    error = 0.0
    for bin_id in range(num_bins):
        selected = bin_ids == bin_id
        if np.any(selected):
            error += float(np.mean(selected)) * abs(
                float(np.mean(predictions[selected])) - float(np.mean(labels[selected]))
            )
    return error


def bootstrap_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    num_resamples: int = 10_000,
    statistic: str = "mean",
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(num_resamples, len(values)))
    sampled = values[indices]
    if statistic == "mean":
        estimates = np.mean(sampled, axis=1)
    elif statistic == "median":
        estimates = np.median(sampled, axis=1)
    else:
        raise ValueError(statistic)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)
