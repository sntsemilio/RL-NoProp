"""Estimate each action's expected reward through stateless incremental learning."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Cada acción tiene una estimación de recompensa esperada, sin estados.
def estimate_action_value(
    num_samples,
    reward_outcomes,
    reward_probabilities,
    constant_step_size=None,
):
    q_value_history = np.zeros(num_samples)
    rewards = np.random.choice(
        a=reward_outcomes,
        p=reward_probabilities,
        size=num_samples,
    )

    for sample_count in range(1, num_samples):
        step_size = (
            1 / sample_count
            if constant_step_size is None
            else constant_step_size
        )
        previous_q_value = q_value_history[sample_count - 1]
        reward = rewards[sample_count]
        q_value_history[sample_count] = previous_q_value + step_size * (
            reward - previous_q_value
        )

    return q_value_history

# Gráficas
def plot_action_value_estimates(
    action_0_q_value_history,
    action_1_q_value_history,
    plot_title,
):
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    final_action_0_q_value = action_0_q_value_history[-1]
    final_action_1_q_value = action_1_q_value_history[-1]

    action_0_has_higher_estimate = (
        final_action_0_q_value > final_action_1_q_value
    )

    # Action 0
    axes[0].plot(
        action_0_q_value_history,
        linewidth=3 if action_0_has_higher_estimate else 1.5,
    )
    axes[0].axhline(final_action_0_q_value, linestyle="--", alpha=0.6)
    axes[0].set_title(
        f"Action 0 | final Q-value = {final_action_0_q_value:.2f}"
    )

    # Action 1
    axes[1].plot(
        action_1_q_value_history,
        linewidth=3 if not action_0_has_higher_estimate else 1.5,
    )
    axes[1].axhline(final_action_1_q_value, linestyle="--", alpha=0.6)
    axes[1].set_title(
        f"Action 1 | final Q-value = {final_action_1_q_value:.2f}"
    )

    figure.suptitle(plot_title, fontsize=13)
    plt.tight_layout()

    output_path = Path(__file__).with_name("bandits.png")
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Graph saved to: {output_path.resolve()}")


# ===== Experimento
NUM_SAMPLES_PER_ACTION = 10000
PLOT_TITLE = "10000 samples per action | Action 0 vs Action 1 | step size = 1/n"

# Action 0
action_0_q_value_history = estimate_action_value(
    NUM_SAMPLES_PER_ACTION,
    reward_outcomes=[10, 0],
    reward_probabilities=[0.40, 0.60],
)

# Action 1
action_1_q_value_history = estimate_action_value(
    NUM_SAMPLES_PER_ACTION,
    reward_outcomes=[100, 0],
    reward_probabilities=[0.03, 0.97],
)

estimated_best_action = (
    "Action 0 has the higher estimate"
    if action_0_q_value_history[-1] > action_1_q_value_history[-1]
    else "Action 1 has the higher estimate"
)
plot_action_value_estimates(
    action_0_q_value_history,
    action_1_q_value_history,
    PLOT_TITLE + " | " + estimated_best_action,
)
