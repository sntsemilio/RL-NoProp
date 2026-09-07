import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from BanditsEnv import BanditEnvironment
from DQN import QNetwork
from ReplayMemory import ReplayBuffer
from TrainingLoop import train_dqn

# -- Config
REPLAY_BUFFER_CAPACITY = 1500

# BANDIT_ARMS = [
#     {"reward_outcomes": [0, 10], "reward_probabilities": [0.6, 0.4]},
#     {"reward_outcomes": [0, 100], "reward_probabilities": [0.97, 0.03]},
# ]

BANDIT_ARMS = [
    {"reward_outcomes": [100, 0], "reward_probabilities": [0.6, 0.4]},
    {"reward_outcomes": [0, 100], "reward_probabilities": [0.4, 0.6]},
]

TARGET_NETWORK_SYNC_INTERVAL = 100

BATCH_SIZE = 32

LEARNING_RATE = 1e-3

NUM_TRAINING_STEPS = 10000
NUM_EVALUATION_STEPS = 10000
STATE_DIM = 1

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995



# -- Initialize environment
training_environment = BanditEnvironment(BANDIT_ARMS)


# -- Initialize online and target network
# Trainable online Q-network
online_network = QNetwork(STATE_DIM, training_environment.num_actions)

# Target Q-network
target_network = QNetwork(STATE_DIM, training_environment.num_actions)

# Load online-network parameters into the target network.
target_network.load_state_dict(online_network.state_dict())

# Set the target network to evaluation mode.
target_network.eval()

# Set optimizer
optimizer = torch.optim.Adam(online_network.parameters(), lr=LEARNING_RATE)


# -- Initialize the Replay Buffer
replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY)


# Execute training loop
q_value_history = train_dqn(
    training_environment,
    online_network,
    target_network,
    replay_buffer,
    NUM_TRAINING_STEPS,
    optimizer,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY,
    batch_size=BATCH_SIZE,
    target_sync_interval=TARGET_NETWORK_SYNC_INTERVAL,
)


#  Graphs
q_value_history = np.array(q_value_history)

plt.figure(figsize=(8, 5))
for action_index in range(q_value_history.shape[1]):
    plt.plot(
        q_value_history[:, action_index],
        label=f"Q(action {action_index})",
    )

plt.xlabel("Training checkpoints")
plt.ylabel("Q-value")
plt.title("DQN learning on bandit")
plt.legend()
plt.tight_layout()
plt.show()



# ------ Test

def greedy_policy(state):
    with torch.no_grad():
        q_values = online_network(state)
        return q_values.argmax(dim=1).item()


def random_policy(_):
    return random.randint(0, training_environment.num_actions - 1)


def evaluate_policy(environment, policy, num_steps=10000):
    total_reward = 0.0
    state = torch.tensor([[1.0]], dtype=torch.float32)

    for _ in range(num_steps):
        action = policy(state)
        reward = environment.step(action)
        total_reward += reward

    return total_reward / num_steps


def evaluate_trained_agent(num_steps=NUM_EVALUATION_STEPS):

    evaluation_bandit_arms = [
        {"reward_outcomes": [200, 0], "reward_probabilities": [0.8, 0.2]},
        {"reward_outcomes": [0, 100], "reward_probabilities": [0.2, 0.8]},
    ]

    # Create test env
    evaluation_environment = BanditEnvironment(evaluation_bandit_arms)

    greedy_mean_reward = evaluate_policy(
        evaluation_environment,
        greedy_policy,
        num_steps,
    )
    random_mean_reward = evaluate_policy(
        evaluation_environment,
        random_policy,
        num_steps,
    )

    print(f"Mean reward (greedy policy): {greedy_mean_reward:.3f}")
    print(f"Mean reward (random policy): {random_mean_reward:.3f}")


evaluate_trained_agent()
