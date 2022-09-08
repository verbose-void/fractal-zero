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
        self.unroll_steps = 8

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

    def _get_batch(self):
        batch = self.data_handler.get_batch(self.unroll_steps)

        (
            self.observations,
            self.actions,
            self.target_auxiliaries,
            self.target_values,
        ) = batch

        return batch

    def _unroll(self):
        # TODO: docstring

        first_observations = self.observations[:, 0]
        first_hidden_states = self.representation_model.forward(first_observations)

        self.dynamics_model.set_state(first_hidden_states)

        # preallocate unroll prediction arrays
        self.unrolled_auxiliaries = torch.zeros_like(self.target_auxiliaries)
        self.unrolled_values = torch.zeros_like(self.target_values)

        # fill arrays
        for unroll_step in range(self.unroll_steps):
            step_actions = self.actions[:, unroll_step]
            self.unrolled_auxiliaries[:, unroll_step] = self.dynamics_model(step_actions)

            state = self.dynamics_model.state
            _, value_predictions = self.prediction_model.forward(state)
            # TODO: unroll policy

            self.unrolled_values[:, unroll_step] = value_predictions

    def _calculate_losses(self):
        auxiliary_loss = self.dynamics_model.auxiliary_loss(
            self.unrolled_auxiliaries, 
            self.target_auxiliaries,
        )

        value_loss = self.prediction_model.value_loss(
            self.unrolled_values, 
            self.target_values
        )

        composite_loss = (auxiliary_loss + value_loss) / self.unroll_steps

        if self.use_wandb:
            wandb.log(
                {
                    "auxiliary_loss": auxiliary_loss.item(),
                    "mean_auxiliary_targets": self.target_auxiliaries.mean(),
                    "value_loss": value_loss.item(),
                    "composite_loss": composite_loss.item(),
                    "mean_value_targets": self.target_values.mean(),
                }
            )

        return composite_loss

    def train_step(self):
        self.fractal_zero.train()

        self.optimizer.zero_grad()

        self._get_batch()
        self._unroll()

        composite_loss = self._calculate_losses()
        composite_loss.backward()

        self.optimizer.step()
