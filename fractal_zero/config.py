from dataclasses import dataclass

import gym
import torch
from fractal_zero.models.joint_model import JointModel

from fractal_zero.utils import get_space_shape


DEFAULT_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@dataclass
class FractalZeroConfig:
    env: gym.Env
    joint_model: JointModel

    max_replay_buffer_size: int = 512
    num_games: int = 5_000
    max_game_steps: int = 200

    max_batch_size: int = 128
    gamma: float = 0.99
    unroll_steps: int = 16
    learning_rate: float = 0.001
    optimizer: str = "SGD"

    num_walkers: int = 64
    balance: float = 1
    lookahead_steps: int = 64
    evaluation_lookahead_steps: int = 64

    device: torch.device = DEFAULT_DEVICE

    wandb_config: dict = None

    @property
    def use_wandb(self) -> bool:
        return self.wandb_config is not None

    @property
    def observation_shape(self) -> tuple:
        return get_space_shape(self.env.observation_space)

    @property
    def action_shape(self) -> tuple:
        return get_space_shape(self.env.action_space)