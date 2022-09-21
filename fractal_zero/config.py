from dataclasses import asdict, dataclass, field
from typing import Callable

import gym
import torch
from torch.optim.lr_scheduler import StepLR
from fractal_zero.models.joint_model import JointModel

from fractal_zero.utils import get_space_shape


# DEFAULT_DEVICE = (
#     torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# )
DEFAULT_DEVICE = torch.device("cpu")


CONSTANT_LR_CONFIG = {
    "alias": "ConstantLR",
    "class": StepLR,
    "step_size": 999999,
    "gamma": 1,
}


@dataclass
class FractalZeroConfig:
    # TODO: break config into multiple parts (FMC, Trainer, etc.)

    env: gym.Env

    # TODO: if using AlphaZero style, autodetermine the embedding size.
    joint_model: JointModel

    # when True the lookahead search uses the environment directly (AlphaZero Style).
    # when False, the lookahead search uses a DynamicsModel instead of the environment (MuZero Style).
    search_using_actual_environment: bool = True

    max_replay_buffer_size: int = 512
    replay_buffer_pop_strategy: str = "oldest"  # oldest or random
    num_games: int = 5_000
    max_game_steps: int = 200

    max_batch_size: int = 128
    dynamic_batch_size: bool = True
    gamma: float = 0.99
    unroll_steps: int = 16
    minimize_batch_padding: bool = True
    learning_rate: float = 0.001
    lr_scheduler_config: dict = field(default_factory=lambda: CONSTANT_LR_CONFIG)
    weight_decay: float = 1e-4
    momentum: float = 0.9  # only if optimizer is SGD
    optimizer: str = "SGD"

    num_walkers: int = 64
    balance: float = 1
    lookahead_steps: int = 64
    evaluation_lookahead_steps: int = 64
    fmc_backprop_strategy: str = "all"  # all, clone_mask, or clone_participants
    fmc_clone_strategy: str = "cumulative_reward"  # predicted_values or cumulative_reward

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
