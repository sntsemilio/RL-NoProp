import math
import random

import matplotlib.pyplot as plt

# ===========================================================
# CONSTANTS
# ===========================================================
BANDIT_ARMS = [
    {"reward_outcomes": [0, 10], "reward_probabilities": [0.6, 0.4]},
    {"reward_outcomes": [0, 100], "reward_probabilities": [0.97, 0.03]},
]

NUM_TRAINING_STEPS = 10000
NUM_EVALUATION_STEPS = 5000

STEP_SIZE = 0.01

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_RATE = 0.001


# ===========================================================
# ENVIRONMENT
# ===========================================================
def sample_reward(action):
    arm = BANDIT_ARMS[action]
    return random.choices(
        arm["reward_outcomes"],
        arm["reward_probabilities"],
    )[0]


# ===========================================================
# Q-LEARNING (TRAINING)
# ===========================================================
def train_q_learning():
    num_actions = len(BANDIT_ARMS)
    q_values = [0.0 for _ in range(num_actions)]
    q_value_history = [[] for _ in range(num_actions)]

    for training_step in range(NUM_TRAINING_STEPS):

        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(
            -EPSILON_DECAY_RATE * training_step
        )

        # Epsilon-greedy policy
        if random.random() < epsilon:
            action = random.randint(0, num_actions - 1)
        else:
            action = q_values.index(max(q_values))

        reward = sample_reward(action)

        # Q-learning update
        q_values[action] += STEP_SIZE * (reward - q_values[action])

        for action_index in range(num_actions):
            q_value_history[action_index].append(q_values[action_index])

    return q_values, q_value_history


# ===========================================================
# POLICY EVALUATION
# ===========================================================
def evaluate_policy(policy, num_steps):
    total_reward = 0.0
    for _ in range(num_steps):
        action = policy()
        total_reward += sample_reward(action)
    return total_reward / num_steps


# ===========================================================
# MAIN
# ===========================================================
q_values, q_value_history = train_q_learning()

# Greedy policy learned from the final Q-values.
def greedy_policy():
    return q_values.index(max(q_values))

# Random baseline policy
def random_policy():
    return random.randint(0, len(BANDIT_ARMS) - 1)

greedy_mean_reward = evaluate_policy(greedy_policy, NUM_EVALUATION_STEPS)
random_mean_reward = evaluate_policy(random_policy, NUM_EVALUATION_STEPS)

print("Learned Q-values:", q_values)
print(f"Mean reward (greedy policy): {greedy_mean_reward:.3f}")
print(f"Mean reward (random policy): {random_mean_reward:.3f}")


# ===========================================================
# PLOT Q-VALUES ONLY
# ===========================================================
plt.figure(figsize=(8, 5))
for action_index in range(len(BANDIT_ARMS)):
    plt.plot(
        q_value_history[action_index],
        label=f"Q(action {action_index})",
    )

plt.xlabel("Training steps")
plt.ylabel("Q-value")
plt.title("Tabular Q-learning on bandit arms")
plt.legend()
plt.tight_layout()
plt.show()
