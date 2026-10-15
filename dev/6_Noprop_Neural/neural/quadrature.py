"""FP64 scalar Gaussian integration with exact breakpoints of two-layer ReLU.

Each fixed context/action makes the logit piecewise affine in z. Split at every
first-layer zero and then every second-layer zero within each first-layer region.
Gauss-Legendre refinement is applied on these smooth pieces, never across kinks.
"""

import math
from functools import lru_cache

import numpy as np


def sigmoid(u):
    return np.exp(-np.logaddexp(0.0, -u))


class NumericBlock:
    def __init__(self, block):
        self.kind, self.index, self.rho = block.kind, block.index, block.rho
        self.w = [x.detach().cpu().double().numpy().copy() for x in block.weights]
        self.b = [x.detach().cpu().double().numpy().copy() for x in block.biases]

    def base(self, c, a):
        onehot = np.eye(2)[a]
        x = np.concatenate((c, onehot), -1)
        if self.kind == "logistic":
            x = np.concatenate(
                [onehot] + [c[..., j : j + 1] * onehot for j in range(c.shape[-1])], -1
            )
        for k, (w, b) in enumerate(zip(self.w, self.b)):
            x = x @ w + b
            if k < len(self.w) - 1:
                x = np.maximum(x, 0)
        return x[..., 0]

    def pieces(self, c, a, low, high):
        """Intervals [l,r] and exact u(z)=s*z+t; all actual network weights."""
        if self.kind in ("direct", "structured", "logistic"):
            intercept = float(self.base(np.asarray(c), np.asarray(a)))
            slope = 0.0
            if self.kind == "structured" and self.index == 2:
                slope = math.sqrt(self.rho) / (1 - self.rho)
                intercept -= self.rho / (2 * (1 - self.rho))
            return (
                np.array([low]),
                np.array([high]),
                np.array([slope]),
                np.array([intercept]),
            )
        x = np.r_[c, np.eye(2)[a]]
        if self.kind == "legacy":
            return (
                np.array([low]),
                np.array([high]),
                self.w[0][0].copy(),
                np.array([self.w[0][1 + a, 0]]),
            )
        slope1 = self.w[0][0]
        offset1 = x @ self.w[0][1:] + self.b[0]
        roots = np.divide(
            -offset1, slope1, out=np.full_like(offset1, np.inf), where=slope1 != 0
        )
        cuts = np.unique(np.r_[low, roots[(roots > low) & (roots < high)], high])
        left, right = cuts[:-1], cuts[1:]
        on1 = ((left + right)[:, None] / 2 * slope1 + offset1) > 0
        slope2 = (on1 * slope1) @ self.w[1]
        offset2 = (on1 * offset1) @ self.w[1] + self.b[1]
        roots2 = np.divide(
            -offset2, slope2, out=np.full_like(offset2, np.inf), where=slope2 != 0
        )
        valid = (roots2 > left[:, None]) & (roots2 < right[:, None])
        allcuts = np.unique(np.r_[cuts, roots2[valid]])
        left, right = allcuts[:-1], allcuts[1:]
        mid = (left + right) / 2
        parent = np.searchsorted(cuts, mid, side="right") - 1
        s2, t2 = slope2[parent], offset2[parent]
        on2 = (mid[:, None] * s2 + t2) > 0
        slope = ((s2 * on2) @ self.w[2])[:, 0]
        offset = ((t2 * on2) @ self.w[2])[:, 0] + self.b[2][0]
        return left, right, slope, offset


@lru_cache(None)
def legendre(order):
    return np.polynomial.legendre.leggauss(order)


def integral(block, c, a, mean=0.0, std=1.0, mu=None, tol=1e-8):
    low, high = mean - 10 * std, mean + 10 * std
    left, right, slope, offset = block.pieces(c, a, low, high)
    # Bound interval length even for noise-independent/structured logits.
    ll, rr, ss, tt = [], [], [], []
    for l, r, s, t in zip(left, right, slope, offset):
        n = max(1, math.ceil((r - l) / std))
        cuts = np.linspace(l, r, n + 1)
        ll.extend(cuts[:-1])
        rr.extend(cuts[1:])
        ss.extend([s] * n)
        tt.extend([t] * n)
    left, right, slope, offset = map(np.asarray, (ll, rr, ss, tt))
    prev = None
    for order in (8, 16, 32, 64, 128):
        nodes, weights = legendre(order)
        z = (left + right)[:, None] / 2 + (right - left)[:, None] / 2 * nodes
        w = (
            (right - left)[:, None]
            / 2
            * weights
            * np.exp(-0.5 * ((z - mean) / std) ** 2)
            / (std * math.sqrt(2 * math.pi))
        )
        logits = z * slope[:, None] + offset[:, None]
        p = sigmoid(logits)
        vals = [np.sum(w * p)]
        if mu is not None:
            ideal_logits = np.full_like(z, math.log(mu / (1 - mu)))
            if block.index == 2:
                ideal_logits += (math.sqrt(block.rho) * z - block.rho / 2) / (
                    1 - block.rho
                )
            q = sigmoid(ideal_logits)
            kl = q * (np.logaddexp(0, -logits) - np.logaddexp(0, -ideal_logits)) + (
                1 - q
            ) * (np.logaddexp(0, logits) - np.logaddexp(0, ideal_logits))
            vals += [np.sum(w * kl), 0.5 * np.sum(w * (p - q) ** 2)]
        vals = np.asarray(vals)
        if prev is not None and np.max(np.abs(vals - prev)) <= tol:
            return vals, float(np.max(np.abs(vals - prev))), order
        prev = vals
    raise ArithmeticError(
        f"Quadrature failed fixed tolerance {tol}, context={c}, action={a}"
    )


def readouts(learner, contexts, actions, tol=1e-8):
    c, a = np.asarray(contexts, dtype=np.float64), np.asarray(actions, dtype=int)
    blocks = [NumericBlock(b) for b in learner.blocks]
    if not learner.composed:
        if blocks[0].kind == "legacy":
            b = blocks[0]
            if b.index == 1:
                p = [integral(b, ci, ai, tol=tol)[0][0] for ci, ai in zip(c, a)]
            else:
                p = sigmoid(
                    b.w[0][0, 0] * math.sqrt(learner.rho) / 2 + b.w[0][1 + a, 0]
                )
            return {"value": np.asarray(p)}, {"max_refinement": 0.0, "max_order": 16}
        ps = [sigmoid(b.base(c, a)) for b in blocks]
        return {"value": np.mean(ps, axis=0)}, {"max_refinement": 0.0, "max_order": 0}
    p1, p2, a0, a1 = [], [], [], []
    err, order = 0.0, 0
    for ci, ai in zip(c, a):
        if blocks[0].kind == "structured":
            p1.append(sigmoid(blocks[0].base(ci, ai)))
        else:
            v, e, o = integral(blocks[0], ci, ai, tol=tol)
            p1.append(v[0])
            err = max(err, e)
            order = max(order, o)
        mid = math.sqrt(learner.rho) / 2
        _, _, s, t = blocks[1].pieces(ci, ai, mid - 1e-10, mid + 1e-10)
        p2.append(sigmoid(s[0] * mid + t[0]))
        for label, output in ((0, a0), (1, a1)):
            v, e, o = integral(
                blocks[1],
                ci,
                ai,
                math.sqrt(learner.rho) * label,
                math.sqrt(1 - learner.rho),
                tol=tol,
            )
            output.append(v[0])
            err = max(err, e)
            order = max(order, o)
    p1, p2, a0, a1 = map(np.asarray, (p1, p2, a0, a1))
    return {
        "value": (1 - p1) * a0 + p1 * a1,
        "d1": p1,
        "d2": p2,
        "same_weights_average": (p1 + p2) / 2,
        "kappa": a1 - a0,
    }, {"max_refinement": err, "max_order": order}
