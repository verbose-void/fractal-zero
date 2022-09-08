import torch

from data.data_handler import DataHandler
from data.replay_buffer import ReplayBuffer
from fractal_zero import FractalZero
from models.joint_model import JointModel

import wandb


class FractalZeroTrainer:
    def __init__(
        self, fractal_zero: FractalZero, data_handler: DataHandler, use_wandb: bool = False
    ):
        self.data_handler = data_handler

        self.fractal_zero = fractal_zero

        # TODO: load from config
        self.optimizer = torch.optim.SGD(self.fractal_zero.parameters(), lr=0.0005)

        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(project="fractal_zero_cartpole")

    @property
    def representation_model(self):
        return self.fractal_zero.model.representation_model

    @property
    def dynamics_model(self):
        return self.fractal_zero.model.dynamics_model

    @property
    def prediction_model(self):
        return self.fractal_zero.model.prediction_model

    def train_step(self):
        self.optimizer.zero_grad()

        (
            observations,
            actions,
            auxiliary_targets,
            value_targets,
        ) = self.data_handler.get_batch()

        hidden_states = self.representation_model(observations)

        # TODO: unroll model from initial hidden state

        self.dynamics_model.set_state(hidden_states)
        auxiliary_predictions = self.dynamics_model(actions)

        policy_logits, value_predictions = self.prediction_model(
            self.dynamics_model.state
        )

        auxiliary_loss = self.dynamics_model.auxiliary_loss(auxiliary_predictions, auxiliary_targets)
        value_loss = self.prediction_model.value_loss(value_predictions, value_targets)
        composite_loss = auxiliary_loss + value_loss
        composite_loss.backward()

        if self.use_wandb:
            wandb.log(
                {
                    "auxiliary_loss": auxiliary_loss.item(),
                    "mean_auxiliary_targets": torch.mean(auxiliary_targets),
                    "value_loss": value_loss.item(),
                    "composite_loss": composite_loss.item(),
                    "mean_value_targets": value_targets.mean(),
                }
            )

        self.optimizer.step()
