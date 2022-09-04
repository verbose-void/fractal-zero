import torch
import numpy as np
import gym

from utils import get_space_shape


class FullyConnectedDynamicsModel(torch.nn.Module):

    def __init__(self, env: gym.Env, embedding_size: int, out_features: int = 2):
        super().__init__()

        self.action_space = env.action_space
        self.action_shape = get_space_shape(self.action_space)

        self.embedding_size = embedding_size
        self.out_features = out_features

        in_dim = self.embedding_size + np.prod(self.action_shape).astype(int)
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
        if len(action.shape) == 0:
            action = torch.unsqueeze(action, 0)

        x = torch.concat((self.state, action))

        self.state = self.embedding_net(x)
        output = self.auxiliary_net(self.state)
        
        return output