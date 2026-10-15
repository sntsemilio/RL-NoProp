"""Oracle-based diagnostics live strictly outside the interacting learner."""

import time

import numpy as np
from neural.core import seed_for
from neural.environment import evaluation_contexts, means
from neural.evaluation import value_metrics
from neural.quadrature import readouts

from .io import npz_write


def decisions(p, mu):
    p, mu = p.reshape(-1, 2), mu.reshape(-1, 2)
    choice, best = np.argmax(p, 1), np.argmax(mu, 1)
    gap = np.abs(mu[:, 0] - mu[:, 1])
    tie = p[:, 0] == p[:, 1]
    wrong = (choice != best).astype(float)
    wrong[tie] = 0.5
    wrong[gap == 0] = 0
    return {
        "choice": choice.astype(np.int8),
        "prediction_tie": tie,
        "gap": gap,
        "wrong_probability": wrong,
        "regret": gap * wrong,
    }


def gap_summary(gap, wrong):
    # Weights are 0, 1/2 or 1, including randomized predicted ties.
    distribution = np.repeat(gap, np.rint(2 * wrong).astype(int))
    return {
        "wrong_fraction_non_tied": float(wrong.sum() / max(1, (gap > 0).sum())),
        "wrong_gap_mean": float(np.average(gap, weights=wrong))
        if wrong.sum()
        else None,
        "wrong_gap_quantiles": dict(
            zip(
                ("q25", "q50", "q75", "q90", "q95", "q99"),
                map(
                    float, np.quantile(distribution, (0.25, 0.5, 0.75, 0.9, 0.95, 0.99))
                ),
            )
        )
        if len(distribution)
        else {},
    }


def diagnostic(learner, job, destination, t):
    start = time.perf_counter()
    cc = evaluation_contexts(job["seed"], job.get("evaluation_contexts", 4096))
    c, a = np.repeat(cc, 2, axis=0), np.tile(np.arange(2), len(cc))
    p, quad = readouts(learner, c, a, tol=1e-8)
    mu = means(c, a, job["family"], job["beta"])
    r = (
        np.random.default_rng(seed_for(job["seed"], "diagnostic_rewards")).random(
            len(a)
        )
        < mu
    ).astype(float)
    metrics = {
        key: {
            **value_metrics(value, mu, r),
            **gap_summary(
                decisions(value, mu)["gap"], decisions(value, mu)["wrong_probability"]
            ),
        }
        for key, value in p.items()
        if key != "kappa"
    }
    npz_write(
        destination / f"evaluation_t{t}.npz",
        contexts=c,
        actions=a,
        mu=mu,
        rewards=r,
        **p,
    )
    return {
        "t": t,
        "metrics": metrics,
        "quadrature": quad,
        "seconds": time.perf_counter() - start,
    }


def trajectory_metrics(c, a, r, family, beta):
    mu = np.stack(
        [means(c.astype(float), np.full(len(c), j), family, beta) for j in range(2)], 1
    )
    best = np.argmax(mu, 1)
    gap = np.abs(mu[:, 0] - mu[:, 1])
    wrong = ((a != best) & (gap > 0)).astype(float)
    regret = mu.max(1) - mu[np.arange(len(a)), a]
    result = dict(
        mean_regret=float(regret.mean()),
        cumulative_regret=float(regret.sum()),
        observed_reward=float(r.mean()),
        interactions=len(a),
        **gap_summary(gap, wrong),
    )
    return result, {"regret": regret, "wrong": wrong, "gap": gap, "mu": mu}
