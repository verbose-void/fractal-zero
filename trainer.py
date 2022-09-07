import torch
import torch.nn.functional as F

from data.data_handler import DataHandler
from data.replay_buffer import ReplayBuffer
from models.joint_model import JointModel

import wandb


class Trainer:
    def __init__(
        self, data_handler: DataHandler, model: JointModel, use_wandb: bool = False
    ):
        self.data_handler = data_handler

        self.model = model

        # TODO: load from config
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)

        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(project="fractal_zero_cartpole")

    def train_step(self):
        self.optimizer.zero_grad()

        (
            observations,
            actions,
            reward_targets,
            value_targets,
        ) = self.data_handler.get_batch()

        hidden_states = self.model.representation_model(observations)

        # TODO: unroll model from initial hidden state

        self.model.dynamics_model.set_state(hidden_states)
        reward_predictions = self.model.dynamics_model(actions)

        policy_logits, value_predictions = self.model.prediction_model(
            self.model.dynamics_model.state
        )

        reward_loss = F.mse_loss(reward_predictions, reward_targets)
        value_loss = F.mse_loss(value_predictions, value_targets)

        cum_loss = reward_loss + value_loss
        cum_loss.backward()

        if self.use_wandb:
            wandb.log(
                {
                    "reward_loss": reward_loss.item(),
                    "mean_reward_targets": torch.mean(reward_targets),
                    "value_loss": value_loss.item(),
                    "cum_loss": cum_loss.item(),
                    "mean_value_targets": value_targets.mean(),
                }
            )

        self.optimizer.step()
