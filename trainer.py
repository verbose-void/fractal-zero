import torch
from data.data_handler import DataHandler
from data.replay_buffer import ReplayBuffer


class Trainer:
    def __init__(self, data_handler: DataHandler, model):
        self.data_handler = data_handler

        self.model = model

        # TODO: load from config
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def train_step(self):
        observations, actions, rewards = self.data_handler.get_batch()
        print(observations.shape, actions.shape, rewards.shape)