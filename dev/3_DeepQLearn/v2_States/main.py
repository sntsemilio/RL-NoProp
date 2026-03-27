from DQN import DQN
from BanditsEnv import BanditsEnv
from ReplayMemory import ReplayMemory
from TrainingLoop import train
import torch, random, datetime
import numpy as np
import matplotlib.pyplot as plt

# -- Config
BUFFER_CAPACITY = 8000
TARGET_UPDATE = 100
BATCH_SIZE = 64
LR = 1e-3
EPISODES = 100000
BLOCK_SIZE = 5000

# -- Initialize environment
ENV = BanditsEnv(block_size=BLOCK_SIZE)

# -- Initialize online and target network
# Trainable Net
onlineNet = DQN(8, ENV.num_actions)

# Target Net
targetNet = DQN(8, ENV.num_actions)

# ------ Test Functions
def greedy_policy(state):
    with torch.no_grad():
        q_values = onlineNet(state)
        return torch.argmax(q_values).item()

def random_policy(_):
    return random.randint(0, ENV.num_actions - 1)

def evaluate(env, policy_fn, steps):
    total_reward = 0.0
    correct = 0

    for _ in range(steps):
        state = env.getState()
        action = policy_fn(state)
        reward = env.step(action)

        total_reward += reward

        # Check if action is correct given the state
        if env.current_state_name == "AB":
            correct_action = 0  # left
        else:
            correct_action = 1  # right

        if action == correct_action:
            correct += 1

    avg_reward = total_reward / steps
    accuracy = correct / steps

    return avg_reward, accuracy

def test(steps = 10000):
    # Create test env
    test_env = BanditsEnv(block_size=BLOCK_SIZE)

    avg_learned, acc_learned = evaluate(test_env, greedy_policy, steps)
    avg_random, acc_random = evaluate(test_env, random_policy, steps)

    print(f"Learned -> Reward: {avg_learned:.3f} | Accuracy: {acc_learned:.3f}")
    print(f"Random  -> Reward: {avg_random:.3f} | Accuracy: {acc_random:.3f}")


def main():

    # Load trainable net parameters into target net 
    targetNet.load_state_dict(onlineNet.state_dict())

    # Set target net into evaluation mode 
    targetNet.eval()

    # Set optimizer
    optimizer = torch.optim.Adam(onlineNet.parameters(), lr=LR)

    # -- Initialize the Replay Buffer
    buffer = ReplayMemory( capacity=BUFFER_CAPACITY )

    # -- Initialize Epsilon-greedy
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995


    # Execute training loop
    q_history = train(
        ENV,
        onlineNet, 
        targetNet,
        buffer,
        EPISODES,
        optimizer,
        epsilon,
        epsilon_min,
        epsilon_decay,
        batch_size = BATCH_SIZE,
        targetNet_update = TARGET_UPDATE
    )

    test(steps=10000)

    # ------ Graphs
    q_history = np.array(q_history)

    plt.figure(figsize=(8, 5))
    for i in range(q_history.shape[1]):
        plt.plot(q_history[:, i], label=f"Q(slot {i})")

    plt.xlabel("Training checkpoints")
    plt.ylabel("Q-value")
    plt.title(f"DQN learning on bandit")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    start = datetime.datetime.now()
    print("\n" + "\033[0;34m" + "[start] " + str(start) + "\033[0m" + "\n");
    main();
    end = datetime.datetime.now()
    print("\n" + "\033[0;34m" + "[end] "+ str(end) + "\033[0m" + "\n");

    exectime= end - start
    print("Exectime: ",exectime.total_seconds() )