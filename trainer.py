import torch
import torch.nn.functional as F

from data.data_handler import DataHandler
from data.replay_buffer import ReplayBuffer
from models.joint_model import JointModel

import wandb


class Trainer:
    def __init__(self, data_handler: DataHandler, model: JointModel):
        self.data_handler = data_handler

        self.model = model

        # TODO: load from config
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        wandb.init(project="fractal_zero_cartpole")

    def train_step(self):
        self.optimizer.zero_grad()

        observations, actions, reward_targets = self.data_handler.get_batch()

        hidden_states = self.model.representation_model(observations)

        self.model.dynamics_model.set_state(hidden_states)
        reward_predictions = self.model.dynamics_model(actions)

        reward_loss = F.mse_loss(reward_predictions, reward_targets)
        reward_loss.backward()

        # print(observations.shape, actions.shape, reward_targets.shape)
        # print(reward_predictions)
        # print(reward_targets)
        # print(reward_loss.item())

        wandb.log({"reward_loss": reward_loss.item(), "mean_reward_targets": torch.mean(reward_targets)})

        self.optimizer.step()