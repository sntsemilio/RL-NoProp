"""Implementación mínima y reproducible de NoProp-RL para bandits Bernoulli."""

from .model import LogisticBlock, compose_mc, corrupt_rewards, stable_sigmoid
from .quadrature import Quadrature

__all__ = [
    "LogisticBlock",
    "Quadrature",
    "compose_mc",
    "corrupt_rewards",
    "stable_sigmoid",
]
