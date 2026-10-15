"""Learners see only contexts, chosen actions and observed real rewards."""

import copy

import numpy as np
import torch
from neural.core import Block, Learner, generator, sample, seed_for


class MSELearner(Learner):
    """Independent direct/ensemble controls, using the audited half-MSE update."""

    def __init__(self, method, seed, lr=0.1, dtype=torch.float32):
        assert method in ("direct_mse", "ensemble_mse")
        super().__init__(
            "ensemble" if method == "ensemble_mse" else "direct", seed, lr, dtype=dtype
        )
        self.method, self.loss_name = method, "mse"
        self.blocks = [
            Block("direct", k + 1, seed_for(seed, method, "init", k), dtype=dtype)
            for k in range(len(self.blocks))
        ]
        self.optimizers = [
            torch.optim.SGD(b.parameters(), lr=lr, momentum=0, weight_decay=0)
            for b in self.blocks
        ]
        self.noise_rng = [
            generator(seed, method, "train_noise", k) for k in range(len(self.blocks))
        ]


@torch.no_grad()
def base_probability(block, c, a):
    """Propensity: run the same weights without the analytic channel offset."""
    x = block.features(c, a, torch.zeros_like(a, dtype=c.dtype))
    for i, (w, b) in enumerate(zip(block.weights, block.biases)):
        x = x @ w + b
        if i < len(block.weights) - 1:
            x = x.relu()
    return x.squeeze(-1).sigmoid()


class InteractiveAgent:
    def __init__(self, method, seed, lr, capacity=20000, device="cpu"):
        self.method, self.seed, self.device = method, seed, device
        self.learner = Learner(
            "structured" if method == "structured_average" else method,
            seed,
            lr,
            device=device,
        )
        self.c = torch.empty((capacity, 2), dtype=torch.float32, device=device)
        self.a = torch.empty(capacity, dtype=torch.int64, device=device)
        self.r = torch.empty(capacity, dtype=torch.float32, device=device)
        self.size = 0
        self.streams = [
            generator(seed, "policy_inference", j, device=device) for j in range(3)
        ]
        self.epsilon_rng = np.random.default_rng(seed_for(seed, "epsilon"))
        self.arm_rng = np.random.default_rng(seed_for(seed, "exploration_arm"))
        self.tie_rng = np.random.default_rng(seed_for(seed, "tie"))
        self.actions = torch.arange(2, device=device)

    @torch.no_grad()
    def values(self, context):
        c = torch.as_tensor(context, dtype=torch.float32, device=self.device).expand(
            2, -1
        )
        if self.method in ("structured", "flexible"):
            return sample(self.learner, c, self.actions, 32, self.streams)
        return torch.stack(
            [base_probability(b, c, self.actions) for b in self.learner.blocks]
        ).mean(0)

    def choose(self, context):
        t = self.size + 1
        if t <= 2:
            return t - 1
        p = self.values(context)
        if not bool(torch.isfinite(p).all()):
            raise FloatingPointError("Nonfinite deployed probability")
        if self.epsilon_rng.random() < t**-0.5:
            return int(self.arm_rng.integers(2))
        values = p.cpu().numpy()
        ties = np.flatnonzero(values == values.max())
        return int(ties[int(self.tie_rng.integers(len(ties)))])

    def observe(self, context, action, reward):
        if self.size >= len(self.r):
            raise RuntimeError("Buffer cap reached")
        self.c[self.size] = torch.as_tensor(context, device=self.device)
        self.a[self.size], self.r[self.size] = action, reward
        self.size += 1
        if self.size < 2:
            return None
        losses = self.learner.update(
            self.c[: self.size], self.a[: self.size], self.r[: self.size], batch=128
        )
        if not bool(torch.isfinite(losses).all()):
            raise FloatingPointError(
                "Nonfinite local loss; no clipping or discarded seed"
            )
        if self.size % 100 == 0 and not all(
            bool(torch.isfinite(p).all())
            for b in self.learner.blocks
            for p in b.parameters()
        ):
            raise FloatingPointError("Nonfinite parameters")
        return losses

    def snapshot(self):
        return copy.deepcopy(
            {
                "learner": self.learner.snapshot(),
                "size": self.size,
                "buffer": [
                    x[: self.size].cpu().clone() for x in (self.c, self.a, self.r)
                ],
                "inference_rng": [g.get_state() for g in self.streams],
                "epsilon_rng": self.epsilon_rng.bit_generator.state,
                "arm_rng": self.arm_rng.bit_generator.state,
                "tie_rng": self.tie_rng.bit_generator.state,
            }
        )

    def restore(self, state):
        self.learner.restore(state["learner"])
        self.size = state["size"]
        for target, source in zip((self.c, self.a, self.r), state["buffer"]):
            target[: self.size] = source.to(self.device)
        for g, s in zip(self.streams, state["inference_rng"]):
            g.set_state(s)
        for name in ("epsilon_rng", "arm_rng", "tie_rng"):
            getattr(self, name).bit_generator.state = state[name]
