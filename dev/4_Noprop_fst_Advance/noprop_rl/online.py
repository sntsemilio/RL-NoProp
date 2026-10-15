"""Flujo online decidir–observar–actualizar con feedback parcial."""

from __future__ import annotations

import os
import pickle
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np

from .model import LogisticBlock, LossName, compose_mc
from .protocol import derived_seed

OnlineMethod = Literal[
    "tabular_mean",
    "two_block_mse",
    "two_block_bce",
    "d1_mse",
    "d2_mse",
    "mean_substitute_mse",
]


@dataclass(frozen=True)
class OnlineConfig:
    horizon: int
    rho: float
    num_inference_replicas: int
    batch_size: int
    updates_per_interaction: int
    learning_rate_mse: float
    learning_rate_bce: float
    balanced_buffer: bool = False
    record_event_log: bool = True


class RewardSequenceEnvironment:
    """El agente ve recompensas consumidas, nunca probabilidades verdaderas."""

    def __init__(self, reward_sequences: np.ndarray) -> None:
        reward_sequences = np.asarray(reward_sequences, dtype=np.int8)
        if reward_sequences.ndim != 2 or reward_sequences.shape[0] != 2:
            raise ValueError("Se requieren dos secuencias de recompensas")
        self._reward_sequences = reward_sequences.copy()
        self.positions = np.zeros(2, dtype=np.int64)

    def step(self, action: int) -> int:
        position = int(self.positions[action])
        if position >= self._reward_sequences.shape[1]:
            raise RuntimeError("Secuencia de recompensas agotada")
        reward = int(self._reward_sequences[action, position])
        self.positions[action] += 1
        return reward

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"reward_sequences": self._reward_sequences, "positions": self.positions.copy()}

    @classmethod
    def from_state_dict(cls, state: dict[str, np.ndarray]) -> RewardSequenceEnvironment:
        environment = cls(state["reward_sequences"])
        environment.positions = np.asarray(state["positions"], dtype=np.int64).copy()
        return environment


def generate_reward_sequences(
    means: tuple[float, float], horizon: int, environment_seed: int
) -> np.ndarray:
    sequences = []
    for action, mean_reward in enumerate(means):
        rng = np.random.default_rng(derived_seed(environment_seed, action))
        sequences.append(rng.binomial(1, mean_reward, horizon))
    return np.asarray(sequences, dtype=np.int8)


class OnlineRunner:
    """Estado completo reanudable. No almacena medias verdaderas ni brechas."""

    _STREAM_LABELS: ClassVar[dict[str, int]] = {
        "exploration": 1,
        "tie_breaking": 2,
        "minibatches": 3,
        "training_noise_1": 4,
        "training_noise_2": 5,
        "inference": 6,
    }

    def __init__(
        self,
        environment: RewardSequenceEnvironment,
        config: OnlineConfig,
        method: OnlineMethod,
        seed: int,
    ) -> None:
        self.environment = environment
        self.config = config
        self.method = method
        self.seed = seed
        self.block_1 = LogisticBlock()
        self.block_2 = LogisticBlock()
        self.buffer_actions: list[int] = []
        self.buffer_rewards: list[int] = []
        self.actions: list[int] = []
        self.rewards: list[int] = []
        self.estimated_values: list[list[float]] = []
        self.losses: list[dict[str, float]] = []
        self.decision_durations_seconds: list[float] = []
        self.training_durations_seconds: list[float] = []
        self.event_log: list[dict[str, object]] = []
        self.synthetic_buffer_insertions = 0
        self.rngs = {
            name: np.random.default_rng(derived_seed(seed, label))
            for name, label in self._STREAM_LABELS.items()
        }

    @property
    def round_number(self) -> int:
        return len(self.actions) + 1

    def _tabular_values(self) -> np.ndarray:
        values = []
        actions = np.asarray(self.buffer_actions)
        rewards = np.asarray(self.buffer_rewards, dtype=np.float64)
        for action in range(2):
            selected = rewards[actions == action]
            values.append(float(np.mean(selected)) if len(selected) else 0.5)
        return np.asarray(values)

    def estimate_values(self) -> np.ndarray:
        if self.method == "tabular_mean":
            return self._tabular_values()
        if self.method == "d1_mse":
            values = []
            for action in range(2):
                actions = np.full(self.config.num_inference_replicas, action, dtype=np.int64)
                noise = self.rngs["inference"].standard_normal(
                    self.config.num_inference_replicas
                )
                values.append(float(np.mean(self.block_1.predict(noise, actions))))
            return np.asarray(values)
        if self.method == "d2_mse":
            z = np.full(2, np.sqrt(self.config.rho) / 2.0)
            return self.block_2.predict(z, np.arange(2, dtype=np.int64))
        mean_substitute = self.method == "mean_substitute_mse"
        return np.asarray(
            [
                compose_mc(
                    self.block_1,
                    self.block_2,
                    action,
                    self.config.rho,
                    self.config.num_inference_replicas,
                    self.rngs["inference"],
                    mean_substitute=mean_substitute,
                )
                for action in range(2)
            ]
        )

    def decide(self) -> tuple[int, np.ndarray, str]:
        round_number = self.round_number
        if round_number <= 2:
            return round_number - 1, np.array([np.nan, np.nan]), "forced_initialization"
        values = self.estimate_values()
        epsilon = round_number**-0.5
        if self.rngs["exploration"].random() < epsilon:
            action = int(self.rngs["exploration"].integers(0, 2))
            decision_type = "exploration"
        else:
            maxima = np.flatnonzero(values == np.max(values))
            if len(maxima) == 1:
                action = int(maxima[0])
                decision_type = "greedy"
            else:
                action = int(self.rngs["tie_breaking"].choice(maxima))
                decision_type = "random_tie_break"
        return action, values, decision_type

    def _sample_minibatch(self) -> tuple[np.ndarray, np.ndarray]:
        actions = np.asarray(self.buffer_actions, dtype=np.int64)
        rewards = np.asarray(self.buffer_rewards, dtype=np.float64)
        if not self.config.balanced_buffer:
            indices = self.rngs["minibatches"].integers(
                0, len(actions), size=self.config.batch_size
            )
            return actions[indices], rewards[indices]
        # En la primera inicialización aún no existe el otro brazo; solo entonces
        # se usa muestreo uniforme. Desde la segunda observación, el lote es balanceado.
        if any(not np.any(actions == action) for action in range(2)):
            indices = self.rngs["minibatches"].integers(
                0, len(actions), size=self.config.batch_size
            )
            return actions[indices], rewards[indices]
        first_size = self.config.batch_size // 2
        counts = [first_size, self.config.batch_size - first_size]
        selected_indices = []
        for action, count in enumerate(counts):
            available = np.flatnonzero(actions == action)
            selected_indices.append(self.rngs["minibatches"].choice(available, count, replace=True))
        indices = np.concatenate(selected_indices)
        permutation = self.rngs["minibatches"].permutation(self.config.batch_size)
        indices = indices[permutation]
        return actions[indices], rewards[indices]

    def update(self) -> dict[str, float]:
        if self.method == "tabular_mean":
            return {}
        loss_name: LossName = "bce" if self.method == "two_block_bce" else "mse"
        learning_rate = (
            self.config.learning_rate_bce if loss_name == "bce" else self.config.learning_rate_mse
        )
        recorded: dict[str, float] = {}
        for _ in range(self.config.updates_per_interaction):
            actions, rewards = self._sample_minibatch()
            if self.method != "d2_mse":
                z1 = self.rngs["training_noise_1"].standard_normal(len(actions))
                recorded["loss_1"] = self.block_1.update(
                    z1, actions, rewards, loss_name, learning_rate
                )
            if self.method != "d1_mse":
                z2 = np.sqrt(self.config.rho) * rewards + np.sqrt(
                    1.0 - self.config.rho
                ) * self.rngs["training_noise_2"].standard_normal(len(actions))
                recorded["loss_2"] = self.block_2.update(
                    z2, actions, rewards, loss_name, learning_rate
                )
        return recorded

    def step_once(self) -> None:
        if len(self.actions) >= self.config.horizon:
            raise RuntimeError("El horizonte ya se completó")
        round_number = self.round_number
        decision_start = time.perf_counter()
        action, values, decision_type = self.decide()
        self.decision_durations_seconds.append(time.perf_counter() - decision_start)
        if self.config.record_event_log:
            self.event_log.append(
                {
                    "round": round_number,
                    "event": "decision",
                    "action": action,
                    "decision_type": decision_type,
                    "buffer_size_before": len(self.buffer_actions),
                }
            )
        reward = self.environment.step(action)
        if self.config.record_event_log:
            self.event_log.append(
                {"round": round_number, "event": "observation", "action": action, "reward": reward}
            )
        self.actions.append(action)
        self.rewards.append(reward)
        self.estimated_values.append(values.tolist())
        self.buffer_actions.append(action)
        self.buffer_rewards.append(reward)
        if self.config.record_event_log:
            self.event_log.append(
                {
                    "round": round_number,
                    "event": "buffer_append",
                    "real_environment_reward": True,
                    "buffer_size_after": len(self.buffer_actions),
                }
            )
        training_start = time.perf_counter()
        recorded_losses = self.update()
        self.training_durations_seconds.append(time.perf_counter() - training_start)
        self.losses.append(recorded_losses)
        if self.config.record_event_log:
            self.event_log.append(
                {
                    "round": round_number,
                    "event": "update",
                    "updates": self.config.updates_per_interaction
                    if self.method != "tabular_mean"
                    else 0,
                    **recorded_losses,
                }
            )

    def run(self, until: int | None = None) -> None:
        target = self.config.horizon if until is None else until
        if target > self.config.horizon:
            raise ValueError("until excede el horizonte")
        while len(self.actions) < target:
            self.step_once()

    def state_dict(self) -> dict[str, object]:
        return {
            "environment": self.environment.state_dict(),
            "config": asdict(self.config),
            "method": self.method,
            "seed": self.seed,
            "block_1": {
                "parameters": self.block_1.parameters,
                "evaluation_count": self.block_1.evaluation_count,
                "update_count": self.block_1.update_count,
            },
            "block_2": {
                "parameters": self.block_2.parameters,
                "evaluation_count": self.block_2.evaluation_count,
                "update_count": self.block_2.update_count,
            },
            "buffer_actions": self.buffer_actions,
            "buffer_rewards": self.buffer_rewards,
            "actions": self.actions,
            "rewards": self.rewards,
            "estimated_values": self.estimated_values,
            "losses": self.losses,
            "decision_durations_seconds": self.decision_durations_seconds,
            "training_durations_seconds": self.training_durations_seconds,
            "event_log": self.event_log,
            "synthetic_buffer_insertions": self.synthetic_buffer_insertions,
            "rng_states": {name: rng.bit_generator.state for name, rng in self.rngs.items()},
        }

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(descriptor, "wb") as output:
                pickle.dump(self.state_dict(), output, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @classmethod
    def from_checkpoint(cls, path: Path) -> OnlineRunner:
        with path.open("rb") as source:
            # Solo se deben cargar checkpoints locales generados por este proyecto.
            state = pickle.load(source)
        runner = cls(
            RewardSequenceEnvironment.from_state_dict(state["environment"]),
            OnlineConfig(**state["config"]),
            state["method"],
            state["seed"],
        )
        for block, key in ((runner.block_1, "block_1"), (runner.block_2, "block_2")):
            block.parameters = np.asarray(state[key]["parameters"], dtype=np.float64).copy()
            block.evaluation_count = state[key]["evaluation_count"]
            block.update_count = state[key]["update_count"]
        for attribute in (
            "buffer_actions",
            "buffer_rewards",
            "actions",
            "rewards",
            "estimated_values",
            "losses",
            "decision_durations_seconds",
            "training_durations_seconds",
            "event_log",
            "synthetic_buffer_insertions",
        ):
            setattr(runner, attribute, state[attribute])
        for name, rng_state in state["rng_states"].items():
            runner.rngs[name].bit_generator.state = rng_state
        return runner


def evaluate_run(actions: list[int], means: tuple[float, float]) -> dict[str, float]:
    """Diagnóstico externo: esta función no es accesible desde OnlineRunner."""

    actions_array = np.asarray(actions, dtype=np.int64)
    means_array = np.asarray(means, dtype=np.float64)
    gap = abs(means[0] - means[1])
    optimal_mean = float(np.max(means_array))
    pseudoregret = float(np.sum(optimal_mean - means_array[actions_array]))
    result = {
        "pseudoregret": pseudoregret,
        "gap": gap,
        "suboptimal_action_fraction": float(np.mean(means_array[actions_array] < optimal_mean)),
    }
    result["normalized_pseudoregret"] = pseudoregret / (len(actions) * gap) if gap > 0 else np.nan
    return result
