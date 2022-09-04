import torch
import numpy as np


class FullyConnectedDynamicsModel(torch.nn.Module):

    def __init__(self, action_shape: tuple, embedding_size: int, out_features: int = 2):
        super().__init__()

        self.action_shape = action_shape
        self.embedding_size = embedding_size
        self.out_features = out_features

        in_dim = self.embedding_size + np.prod(action_shape).astype(int)
        self.embedding_net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, self.embedding_size),
        )

        self.auxiliary_net = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.out_features)
        )

        self.state = None
    
    def set_state(self, state: torch.Tensor):
        self.state = state

    def forward(self, action):
        x = torch.concat(self.state, action)

        self.state = self.embedding_net(x)
        output = self.auxiliary_net(self.state)
        
        return output 