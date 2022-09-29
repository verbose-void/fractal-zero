from typing import Callable, Union
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
        action_vectorizer: Callable,
    ):
        self.env = load_environment(env)
        self.policy_model = policy_model
        self.action_vectorizer = action_vectorizer

    def sample_trajectory(self, max_steps: int = None):
        obs = self.env.reset()

        observations = []
        actions = []

        i = 0
        while True:
            i += 1

            observations.append(torch.tensor(obs, dtype=float))

            action = self.policy_model(obs)
            obs, reward, done, info = self.env.step(action)

            vec_action = self.action_vectorizer(action)
            actions.append(vec_action)

            if done:
                break
            if max_steps is not None and i >= max_steps:
                break

        assert len(observations) == len(actions)

        x = torch.stack(observations)
        t = torch.tensor(actions, dtype=float)

        return x, t