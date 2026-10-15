"""Private simulator. No oracle import in agents.py; no full-feedback API."""

import copy

import numpy as np
from neural.core import seed_for
from neural.environment import means


class PrivateEnvironment:
    def __init__(self, seed, family, beta):
        self.family, self.beta = family, beta
        self.context_rng = np.random.default_rng(
            seed_for(seed, family, beta, "online_context")
        )
        self.reward_rng = np.random.default_rng(
            seed_for(seed, family, beta, "potential_rewards")
        )
        self.t = 0
        self.pending = None

    def context(self):
        if self.pending is not None:
            raise RuntimeError("Previous context requires exactly one chosen action")
        self.pending = self.context_rng.uniform(-1, 1, 2).astype(np.float32)
        return self.pending.copy()

    def step(self, action):
        if self.pending is None or action not in (0, 1):
            raise RuntimeError("Invalid interaction")
        # Both private uniforms are consumed, independent of chosen action.
        uniforms = self.reward_rng.random(2)
        probability = means(
            self.pending[None].astype(float), np.array([action]), self.family, self.beta
        )[0]
        self.t += 1
        self.pending = None
        return float(uniforms[action] < probability)

    def snapshot(self):
        assert self.pending is None
        return copy.deepcopy(
            {
                "t": self.t,
                "context_rng": self.context_rng.bit_generator.state,
                "reward_rng": self.reward_rng.bit_generator.state,
            }
        )

    def restore(self, state):
        self.t = state["t"]
        self.context_rng.bit_generator.state = state["context_rng"]
        self.reward_rng.bit_generator.state = state["reward_rng"]
        self.pending = None
