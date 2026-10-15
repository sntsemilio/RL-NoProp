"""Evaluations outside the learner, including oracle diagnostics and attribution."""

import math
import time

import numpy as np
import torch

from .core import generator, sample, seed_for
from .environment import evaluation_contexts, means
from .quadrature import NumericBlock, integral, readouts


def bce(p, r):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -r * np.log(p) - (1 - r) * np.log1p(-p)


def ece(p, r):
    bins = np.minimum((p * 10).astype(int), 9)
    return sum(
        np.mean(bins == i) * abs(np.mean(p[bins == i]) - np.mean(r[bins == i]))
        for i in range(10)
        if np.any(bins == i)
    )


def value_metrics(p, mu, rewards):
    p, mu, rewards = (x.reshape(-1, 2) for x in (p, mu, rewards))
    error = np.max(abs(p - mu), axis=1)
    choices = np.argmax(p, axis=1)
    best = np.argmax(mu, axis=1)
    gap = abs(mu[:, 0] - mu[:, 1])
    wrong = (choices != best).astype(float)
    wrong[p[:, 0] == p[:, 1]] = 0.5
    wrong[gap == 0] = 0
    boundary = gap <= 0.1
    return {
        "primary": float(error.mean()),
        "value_excess_bce": float(np.mean(bce(p, mu) - bce(mu, mu))),
        "value_excess_mse": float(0.5 * np.mean((p - mu) ** 2)),
        "classification_error": float(wrong.mean()),
        "regret": float(np.mean(gap * wrong)),
        "ece": float(ece(p.flatten(), rewards.flatten())),
        "quantiles": dict(
            zip(
                ["q50", "q90", "q95", "q99"],
                map(float, np.quantile(error, [0.5, 0.9, 0.95, 0.99])),
            )
        ),
        "boundary_count": int(boundary.sum()),
        "boundary_error": float(error[boundary].mean()) if boundary.any() else None,
        "boundary_regret": float((gap * wrong)[boundary].mean())
        if boundary.any()
        else None,
    }


def local_diagnostics(learner, c, a, mu, tol=1e-8):
    output = {}
    for k, block in enumerate(learner.blocks):
        b = NumericBlock(block)
        risks = []
        for ci, ai, mui in zip(c, a, mu):
            if k == 0:
                v, _, _ = integral(b, ci, ai, mu=float(mui), tol=tol)
            else:
                v0, _, _ = integral(
                    b, ci, ai, 0, math.sqrt(1 - learner.rho), mu=float(mui), tol=tol
                )
                v1, _, _ = integral(
                    b,
                    ci,
                    ai,
                    math.sqrt(learner.rho),
                    math.sqrt(1 - learner.rho),
                    mu=float(mui),
                    tol=tol,
                )
                v = (1 - mui) * v0 + mui * v1
            risks.append(v[1:])
        output[f"block_{k + 1}"] = {
            "excess_bce": float(np.mean(risks, axis=0)[0]),
            "excess_mse": float(np.mean(risks, axis=0)[1]),
        }
        rng = np.random.default_rng(seed_for(learner.seed, "denoiser_calibration", k))
        r = (rng.random(len(mu)) < mu).astype(np.float32)
        z = rng.standard_normal(len(mu)).astype(np.float32)
        if k == 1:
            z = math.sqrt(learner.rho) * r + math.sqrt(1 - learner.rho) * z
        with torch.no_grad():
            p = (
                block(
                    torch.as_tensor(c, dtype=learner.dtype, device=learner.device),
                    torch.as_tensor(a, device=learner.device),
                    torch.as_tensor(z, device=learner.device, dtype=learner.dtype),
                )
                .sigmoid()
                .cpu()
                .numpy()
            )
        output[f"block_{k + 1}"]["denoiser_ece"] = float(ece(p, r))
    return output


def evaluate(learner, config, full=True):
    if config["stage"] == "N1":
        c = np.empty((2, 0))
        a = np.arange(2)
        mu = np.array(config["instance"])
    else:
        cc = evaluation_contexts(config["seed"], 4096)
        c = np.repeat(cc, 2, axis=0)
        a = np.tile(np.arange(2), len(cc))
        mu = means(c, a, config["family"], config["beta"])
    t = time.perf_counter()
    pred, quad = readouts(learner, c, a)
    qt = time.perf_counter() - t
    rng = np.random.default_rng(seed_for(config["seed"], "value_calibration"))
    rewards = (rng.random(len(mu)) < mu).astype(float)
    metrics = {
        name: value_metrics(p, mu, rewards)
        for name, p in pred.items()
        if name != "kappa"
    }
    details = {}
    if learner.composed and full:
        t = time.perf_counter()
        details["local"] = local_diagnostics(learner, c[:512], a[:512], mu[:512])
        details["local_evaluation_seconds"] = time.perf_counter() - t
        e1, e2 = pred["d1"] - mu, pred["d2"] - mu
        details["error_correlation"] = (
            float(np.corrcoef(e1, e2)[0, 1])
            if np.std(e1) > 0 and np.std(e2) > 0
            else None
        )
        details["opposite_error_sign_fraction"] = float(np.mean(e1 * e2 < 0))
        details["kappa_mean"] = float(pred["kappa"].mean())
        if learner.method.startswith("structured"):
            details["identity_max_error"] = float(
                np.max(
                    abs(
                        pred["value"]
                        - (
                            pred["kappa"] * pred["d1"]
                            + (1 - pred["kappa"]) * pred["d2"]
                        )
                    )
                )
            )
    # Steady deployment latency: both actions of a single context, M=32.
    tc = torch.as_tensor(c[:2], dtype=learner.dtype, device=learner.device)
    ta = torch.as_tensor(a[:2], device=learner.device)
    timings = []
    for rep in range(6):
        streams = [
            generator(config["seed"], "inference_timing", rep, j, device=learner.device)
            for j in range(3)
        ]
        if learner.device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        if learner.composed:
            sample(learner, tc, ta, 32, streams)
        else:
            with torch.no_grad():
                for b in learner.blocks:
                    b(tc, ta, torch.zeros_like(ta, dtype=learner.dtype)).sigmoid()
        if learner.device == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - t)
    details["inference_seconds_two_actions_M32_median"] = float(np.median(timings[1:]))
    details["inference_warmup_seconds"] = timings[0]
    return {
        "metrics": metrics,
        "diagnostics": details,
        "quadrature": quad,
        "quadrature_seconds": qt,
    }, {"contexts": c, "actions": a, "mu": mu, "rewards": rewards, **pred}
