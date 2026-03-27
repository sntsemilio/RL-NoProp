import random
import torch


class BanditsEnv:
    def __init__(self, block_size=100):
        """
        slots: list of dicts with keys:
            - 'outcomes'
            - 'probabilities'
        """
        self.num_actions = 2
        self.block_size = block_size

        self.states = {
            "AB": torch.tensor([1,0,0,0, 0,1,0,0], dtype=torch.float32),
            "BA": torch.tensor([0,1,0,0, 1,0,0,0], dtype=torch.float32)
        }

        self.current_state_name = "AB"
        self.counter = 0

    # Return the current state (AB or BA)
    def getState(self):
        # Use the current state
        return self.states[self.current_state_name]
        
    def step(self, action):
        # The reward depends on:
        # 1. The current state (AB or BA)
        # 2. The chosen action (left=0, right=1)

        if self.current_state_name == "AB":
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
        self.counter += 1

        # After block_size steps, switch to the other state
        if self.counter >= self.block_size:
            self.counter = 0
            self.current_state_name = "BA" if self.current_state_name == "AB" else "AB"

        return reward