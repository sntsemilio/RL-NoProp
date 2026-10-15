"""S0/S1 contracts and DP reference only. No sequential learning launched."""

import numpy as np
import torch
from neural.future import project_target


def kernel():
    p = np.zeros((5, 2, 5))
    reward = np.zeros((5, 2, 5))
    for s in (1, 2, 3):
        for a, direction in enumerate((-1, 1)):
            p[s, a, s + direction] = 0.9
            p[s, a, s - direction] = 0.1
        reward[s, :, 0] = -1
        reward[s, :, 4] = 1
    # Terminals have no continuation or repeated entry reward.
    return p, reward


def policy_reference(policy=None, gamma=0.95):
    p, r = kernel()
    policy = np.full((5, 2), 0.5) if policy is None else np.asarray(policy)
    transition = np.einsum("sa,san->sn", policy, p)
    rewards = np.einsum("sa,san,san->s", policy, p, r)
    v = np.linalg.solve(np.eye(5) - gamma * transition, rewards)
    q = np.sum(p * (r + gamma * v[None, None, :]), 2)
    residual = float(np.max(abs(v - np.sum(policy * q, 1))))
    assert residual < 1e-10
    return {
        "values": v.tolist(),
        "action_values": q.tolist(),
        "residual": residual,
        "gamma": gamma,
    }


@torch.no_grad()
def frozen_bellman_targets(reward, terminated, next_action_mass, frozen_policy, atoms):
    """Fixed policy, frozen distributional objective: no policy maximization."""
    if next_action_mass.requires_grad or frozen_policy.requires_grad:
        raise ValueError("Freeze both policy and target mass before local fitting")
    mass = (next_action_mass * frozen_policy[..., None]).sum(1)
    return project_target(reward, terminated, mass, atoms, 0.95)


def plan():
    return {
        "status": "PREPARED_NOT_EXECUTED",
        "states": list(range(5)),
        "initial": 2,
        "terminals": [0, 4],
        "gamma": 0.95,
        "slip": 0.1,
        "atoms": np.linspace(-1, 1, 51).tolist(),
        "rewards": "-1 entering0, +1 entering4, otherwise0; terminal value zero after entry",
        "reference": policy_reference(),
        "S0": {
            "policy": "fixed uniform",
            "episodes": 1000,
            "seeds": 30,
            "targets": "real Monte Carlo discounted returns, gamma=.95; no bootstrap",
            "purpose": "local categorical learning and held-out fixed-policy distribution/value calibration",
        },
        "S1": {
            "transitions": 10000,
            "coverage": "uniform over six interior state-action pairs; stochastic kernel preserved",
            "targets": "frozen categorical Bellman targets; fixed reference policy first, frozen greedy-policy variant separately",
            "target_mass": "detached categorical projection; terminal transitions ignore continuation",
            "purpose": "fit fixed Bellman objectives before interleaved target updates",
        },
        "models": "direct categorical, two disjoint structured blocks, same-checkpoint propensity average; comparison chosen after N3",
        "selection": "reserved sequential pilot with equal method budgets; not selected using DP/test",
        "pending": "sequential pilot, rate/seed freeze and training; no S0/S1 learning executed; S2/S3 and Atari pending",
    }
