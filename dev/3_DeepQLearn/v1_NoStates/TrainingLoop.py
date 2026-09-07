import random

import torch
import torch.nn.functional as F


# Bandits-DQN Training Loop
def train_dqn(
    environment,
    online_network,
    target_network,
    replay_buffer,
    num_training_steps,
    optimizer,
    epsilon_start,
    epsilon_end,
    epsilon_decay,
    batch_size=32,
    target_sync_interval=100,
):
    q_value_history = []
    epsilon = epsilon_start

    # Training Loop
    for training_step in range(1, num_training_steps + 1):

        # Select action with epsilon-greedy
        if random.random() < epsilon:
            # Select random action: Exploration
            action = random.randrange(environment.num_actions)
        else:
            # Select action with highest Q value
            with torch.no_grad():
                action_q_values = online_network(
                    torch.tensor(environment.state).unsqueeze(0)
                )
                action = torch.argmax(action_q_values).item()

        # Execute action
        reward = environment.step(action)

        # Store experience in buffer
        replay_buffer.push((environment.state, action, reward))


        # Update the online network once the replay buffer has enough data.
        if len(replay_buffer) >= batch_size:
            # Sample a random transition batch from the replay buffer.
            transition_batch = replay_buffer.sample(batch_size)

            sampled_states, sampled_actions, sampled_rewards = zip(
                *transition_batch
            )

            state_batch = torch.tensor(sampled_states, dtype=torch.float32)
            action_batch = torch.tensor(
                sampled_actions,
                dtype=torch.int64,
            ).unsqueeze(1)
            reward_batch = torch.tensor(
                sampled_rewards,
                dtype=torch.float32,
            ).unsqueeze(1)

            predicted_q_values = online_network(state_batch).gather(
                1,
                action_batch,
            )

            # Bandit target: there is no bootstrapped future return.
            q_targets = reward_batch

            # Loss
            q_loss = F.mse_loss(predicted_q_values, q_targets)

            # Optimization
            optimizer.zero_grad()
            q_loss.backward()
            optimizer.step()

            if training_step % 10 == 0:
                with torch.no_grad():
                    checkpoint_q_values = online_network(
                        torch.tensor(
                            environment.state,
                            dtype=torch.float32,
                        ).unsqueeze(0)
                    )
                    q_value_history.append(
                        checkpoint_q_values.squeeze(0).cpu().numpy()
                    )


        # Update target network
        if training_step % target_sync_interval == 0:
            target_network.load_state_dict(online_network.state_dict())

        # Update epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return q_value_history
