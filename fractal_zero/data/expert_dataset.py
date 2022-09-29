from typing import Union
import torch
import gym

from fractal_zero.vectorized_environment import load_environment


class ExpertDataset:

    def sample_trajectory(self, max_steps: int = None):
        raise NotImplementedError


class ExpertDatasetGenerator(ExpertDataset):
    def __init__(
        self, 
        policy_model, 
        env: Union[str, gym.Env],
    ):
        self.env = load_environment(env)
        self.policy_model = policy_model

    def sample_trajectory(self, max_steps: int = None):
        obs = self.env.reset()

        trajectory = []

        i = 0
        while True:
            i += 1

            action = self.policy_model(obs)
            obs, reward, done, info = self.env.step(action)

            trajectory.append((obs, action))

            if done:
                break
            if max_steps is not None and i >= max_steps:
                break

        return trajectory