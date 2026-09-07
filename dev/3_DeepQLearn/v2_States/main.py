import datetime
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from BanditsEnv import BanditEnvironment
from DQN import QNetwork
from ReplayMemory import ReplayBuffer
from TrainingLoop import train_dqn

# -- Config
REPLAY_BUFFER_CAPACITY = 8000
TARGET_NETWORK_SYNC_INTERVAL = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_TRAINING_STEPS = 100000
NUM_EVALUATION_STEPS = 10000
STATE_DURATION_STEPS = 5000
STATE_DIM = 8
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

# -- Initialize environment
training_environment = BanditEnvironment(
    state_duration_steps=STATE_DURATION_STEPS
)

# -- Initialize online and target network
# Trainable online Q-network
online_network = QNetwork(STATE_DIM, training_environment.num_actions)

# Target Q-network
target_network = QNetwork(STATE_DIM, training_environment.num_actions)

# ------ Test Functions
def greedy_policy(state):
    with torch.no_grad():
        q_values = online_network(state)
        return torch.argmax(q_values).item()


def random_policy(_):
    return random.randint(0, training_environment.num_actions - 1)


def evaluate_policy(environment, policy, num_steps):
    total_reward = 0.0
    optimal_action_count = 0

    for _ in range(num_steps):
        state = environment.get_state()
        action = policy(state)
        reward = environment.step(action)

        total_reward += reward

        # Check whether the selected action is optimal for the current state.
        if environment.current_state_key == "AB":
            optimal_action = 0  # left
        else:
            optimal_action = 1  # right

        if action == optimal_action:
            optimal_action_count += 1

    mean_reward = total_reward / num_steps
    optimal_action_rate = optimal_action_count / num_steps

    return mean_reward, optimal_action_rate


def evaluate_trained_agent(num_steps=NUM_EVALUATION_STEPS):
    # Create test env
    evaluation_environment = BanditEnvironment(
        state_duration_steps=STATE_DURATION_STEPS
    )

    greedy_mean_reward, greedy_optimal_action_rate = evaluate_policy(
        evaluation_environment,
        greedy_policy,
        num_steps,
    )
    random_mean_reward, random_optimal_action_rate = evaluate_policy(
        evaluation_environment,
        random_policy,
        num_steps,
    )

    print(
        "Greedy policy -> "
        f"mean reward: {greedy_mean_reward:.3f} | "
        f"optimal action rate: {greedy_optimal_action_rate:.3f}"
    )
    print(
        "Random policy -> "
        f"mean reward: {random_mean_reward:.3f} | "
        f"optimal action rate: {random_optimal_action_rate:.3f}"
    )


def main():

    # Load online-network parameters into the target network.
    target_network.load_state_dict(online_network.state_dict())

    # Set the target network to evaluation mode.
    target_network.eval()

    # Set optimizer
    optimizer = torch.optim.Adam(
        online_network.parameters(),
        lr=LEARNING_RATE,
    )

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

    evaluate_trained_agent()

    # ------ Graphs
    q_value_history = np.array(q_value_history)

    plt.figure(figsize=(8, 5))
    for action_index in range(q_value_history.shape[1]):
        plt.plot(
            q_value_history[:, action_index],
            label=f"Q(action {action_index})",
        )

    plt.xlabel("Training checkpoints")
    plt.ylabel("Q-value")
    plt.title("DQN learning on contextual bandit")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    start_time = datetime.datetime.now(datetime.UTC).astimezone()
    print("\n" + "\033[0;34m" + "[start] " + str(start_time) + "\033[0m" + "\n")
    main()
    end_time = datetime.datetime.now(datetime.UTC).astimezone()
    print("\n" + "\033[0;34m" + "[end] " + str(end_time) + "\033[0m" + "\n")

    execution_time = end_time - start_time
    print("Execution time: ", execution_time.total_seconds())
