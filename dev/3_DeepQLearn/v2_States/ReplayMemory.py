import random
from collections import deque


class ReplayBuffer:

    # Initialize the replay buffer with the requested capacity.
    def __init__(self, capacity):
        self.transitions = deque(maxlen=capacity)

    # Save a transition
    def push(self, transition):
        self.transitions.append(transition)

    # Get a random transition batch from the replay buffer.
    def sample(self, batch_size):
        sampled_transitions = random.sample(self.transitions, batch_size)
        return sampled_transitions

    # Return the number of stored transitions.
    def __len__(self):
        return len(self.transitions)
