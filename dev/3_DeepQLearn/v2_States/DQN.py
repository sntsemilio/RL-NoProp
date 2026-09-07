import torch.nn.functional as F
from torch import nn


class QNetwork(nn.Module):

    def __init__(self, state_dim, num_actions):
        """
        Parameters:
            state_dim: int
                The number of features in one state vector.
            num_actions: int
                The number of actions we want to predict.
        """
        super().__init__()

        self.hidden_layer_1 = nn.Linear(state_dim, 32)
        self.hidden_layer_2 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, num_actions)

    def forward(self, state_batch):
        """
        Parameters:
            state_batch: torch.Tensor
                A batch of state vectors.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        hidden_activations = F.relu(self.hidden_layer_1(state_batch))
        hidden_activations = F.relu(self.hidden_layer_2(hidden_activations))
        q_values = self.output_layer(hidden_activations)

        return q_values
