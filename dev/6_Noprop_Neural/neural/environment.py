"""Private simulator/evaluator functions; never imported by core.py."""

import numpy as np

from .core import seed_for
from .quadrature import sigmoid

FAMILIES = ("linear", "interaction", "sine", "radial")


def means(c, a, family, beta=2.0):
    h = {
        "linear": lambda: c[:, 0],
        "interaction": lambda: 2 * c[:, 0] * c[:, 1],
        "sine": lambda: np.sin(np.pi * c[:, 0]) + 0.5 * c[:, 1],
        "radial": lambda: c[:, 0] ** 2 + c[:, 1] ** 2 - 0.5,
    }[family]()
    return sigmoid(0.3 * c[:, 1] + beta * (1 - 2 * a) * h)


def triples(seed, family, beta, n, split, instance=None):
    rng = np.random.default_rng(
        seed_for(seed, family, beta, n if split == "train" else "independent", split)
    )
    if instance is not None:
        c = np.empty((2 * n, 0), dtype=np.float32)
        a = np.repeat(np.arange(2), n)
        p = np.asarray(instance)[a]
    else:
        c = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
        a = rng.integers(2, size=n)
        p = means(c.astype(float), a, family, beta)
    r = (rng.random(len(a)) < p).astype(np.float32)
    return c, a, r


def evaluation_contexts(seed, n=4096, dim=2):
    return np.random.default_rng(seed_for(seed, "evaluation_contexts")).uniform(
        -1, 1, (n, dim)
    )
