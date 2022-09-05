import torch
from replay_buffer import ReplayBuffer


class Trainer:
    def __init__(self, replay_buffer: ReplayBuffer, model):
        self.replay_buffer = replay_buffer

        self.model = model

        # TODO: load from config
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def train_step(self):
        pass