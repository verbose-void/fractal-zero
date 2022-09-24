from typing import Union
import torch
import torch.nn.functional as F
import gym
from fractal_zero.search.fmc import FMC

from fractal_zero.vectorized_environment import RayVectorizedEnvironment, load_environment


class OnlineFMCPolicyTrainer:
    """Trains a policy model in an online manner using FMC as the data generator. "Online" means that the latest policy
    weights are used during the search process. So the data is generated, the model is trained, and the cycle continues.
    """

    def __init__(self, env: Union[str, gym.Env], policy_model: torch.nn.Module, num_walkers: int):
        self.env = load_environment(env)
        self.vec_env = RayVectorizedEnvironment(env, num_walkers)
        self.policy_model = policy_model
        

    def train_epsiode(self, max_steps: int):
        self.vec_env.batch_reset()

        self.fmc = FMC(self.vec_env, self.policy_model)
        self.fmc.simulate(max_steps)

        self.fmc.game_histories

        print(self.fmc.visit_buffer)
        game_history_weights = F.softmax(self.fmc.visit_buffer.float(), dim=0)
        print(game_history_weights)
