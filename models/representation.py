import torch
import numpy as np


class FullyConnectedRepresentationModel(torch.nn.Module):
    def __init__(self, observation_shape: tuple, embedding_size: int):
        super().__init__()

        in_dim = np.prod(observation_shape).astype(int)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, embedding_size)
        )

    def forward(self, observation):
        return self.net(observation)