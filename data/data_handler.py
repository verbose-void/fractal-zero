from typing import Tuple
import gym
import torch
import numpy as np
from data.replay_buffer import ReplayBuffer
from utils import get_space_shape


class DataHandler:

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer):
        self.replay_buffer = replay_buffer

        # TODO: config
        self.batch_size = 8

        self.observation_shape = get_space_shape(env.observation_space)
        self.action_shape = get_space_shape(env.action_space)

        # TODO: expert dataset

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observations = np.zeros((self.batch_size, *self.observation_shape), dtype=float)
        actions = np.zeros((self.batch_size, *self.action_shape), dtype=float)
        rewards = np.zeros((self.batch_size, 1), dtype=float)
        
        for i in range(self.batch_size):
            observation, action, reward = self.replay_buffer.sample()

            observations[i] = observation
            actions[i] = action
            rewards[i] = reward

        return torch.tensor(observations).float(), torch.tensor(actions).float(), torch.tensor(rewards).float()