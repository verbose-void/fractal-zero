import torch
from data.data_handler import DataHandler
from data.replay_buffer import ReplayBuffer
from models.joint_model import JointModel


class Trainer:
    def __init__(self, data_handler: DataHandler, model: JointModel):
        self.data_handler = data_handler

        self.model = model

        # TODO: load from config
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def train_step(self):
        observations, actions, rewards = self.data_handler.get_batch()

        hidden_states = self.model.representation_model(observations)

        self.model.dynamics_model.set_state(hidden_states)
        reward_predictions = self.model.dynamics_model(actions)

        print(observations.shape, actions.shape, rewards.shape)
        print(reward_predictions)
        print(rewards)