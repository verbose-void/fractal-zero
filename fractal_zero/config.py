from dataclasses import asdict, dataclass
from typing import Callable

import gym
import torch
from torch.optim.lr_scheduler import StepLR
from fractal_zero.models.joint_model import JointModel

from fractal_zero.utils import get_space_shape


DEFAULT_DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

def _get_constant_lr_scheduler(optimizer):
    return StepLR(optimizer, step_size=999999, gamma=1)


@dataclass
class FractalZeroConfig:
    # TODO: break config into multiple parts (FMC, Trainer, etc.)

    env: gym.Env
    joint_model: JointModel

    max_replay_buffer_size: int = 512
    replay_buffer_pop_strategy: str = "random"  # random or first
    num_games: int = 5_000
    max_game_steps: int = 200

    max_batch_size: int = 128
    dynamic_batch_size: bool = True
    gamma: float = 0.99
    unroll_steps: int = 16
    minimize_batch_padding: bool = True
    learning_rate: float = 0.001
    lr_scheduler_factory: Callable = _get_constant_lr_scheduler
    weight_decay: float = 1e-4
    momentum: float = 0.9  # only if optimizer is SGD
    optimizer: str = "SGD"

    num_walkers: int = 64
    balance: float = 1
    lookahead_steps: int = 64
    evaluation_lookahead_steps: int = 64
    fmc_backprop_strategy: str = "all"  # all, clone_mask, or clone_participants

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

    def asdict(self) -> dict:
        d = asdict(self)

        del d["env"]
        d["env"] = self.env.unwrapped.spec.id

        return d
