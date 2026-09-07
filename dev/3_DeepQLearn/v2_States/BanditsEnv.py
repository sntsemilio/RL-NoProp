import random

import torch


class BanditEnvironment:
    def __init__(self, state_duration_steps=100):
        """
        state_duration_steps: number of interactions before switching state.
        """
        self.num_actions = 2
        self.state_duration_steps = state_duration_steps

        self.state_vectors = {
            "AB": torch.tensor([1, 0, 0, 0, 0, 1, 0, 0], dtype=torch.float32),
            "BA": torch.tensor([0, 1, 0, 0, 1, 0, 0, 0], dtype=torch.float32),
        }

        self.current_state_key = "AB"
        self.steps_in_current_state = 0

    # Return the current state (AB or BA)
    def get_state(self):
        # Use the current state
        return self.state_vectors[self.current_state_key]
        
    def step(self, action):
        # The reward depends on:
        # 1. The current state (AB or BA)
        # 2. The chosen action (left=0, right=1)

        if self.current_state_key == "AB":
            # In AB:
            # left  -> 80% reward
            # right -> 20% reward
            if action == 0:
                reward = random.choices([100, 0], [0.8, 0.2])[0]
            else:
                reward = random.choices([100, 0], [0.2, 0.8])[0]

        else:  # BA
            # In BA:
            # left  -> 20% reward
            # right -> 80% reward
            if action == 0:
                reward = random.choices([100, 0], [0.2, 0.8])[0]
            else:
                reward = random.choices([100, 0], [0.8, 0.2])[0]

        # Count how many steps we have stayed in this state
        self.steps_in_current_state += 1

        # Switch state after the configured number of interactions.
        if self.steps_in_current_state >= self.state_duration_steps:
            self.steps_in_current_state = 0
            self.current_state_key = (
                "BA" if self.current_state_key == "AB" else "AB"
            )

        return reward
