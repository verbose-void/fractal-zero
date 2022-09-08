from typing import Tuple
import gym
import torch
import numpy as np
from data.replay_buffer import ReplayBuffer
from utils import get_space_shape


class DataHandler:
    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer, device, batch_size: int=8):
        self.replay_buffer = replay_buffer
        self.device = device

        # TODO: config
        self.batch_size = batch_size

        self.observation_shape = get_space_shape(env.observation_space)
        self.action_shape = get_space_shape(env.action_space)

        # TODO: expert dataset

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observations = np.zeros((self.batch_size, *self.observation_shape), dtype=float)
        actions = np.zeros((self.batch_size, *self.action_shape), dtype=float)
        auxiliaries = np.zeros((self.batch_size, 1), dtype=float)
        values = np.zeros((self.batch_size, 1), dtype=float)

        for i in range(self.batch_size):
            observation, action, reward, value = self.replay_buffer.sample()

            observations[i] = observation
            actions[i] = action
            auxiliaries[i] = reward  # auxiliary is a generalization of reward.
            values[i] = value

        # TODO: put these on the correct device sooner?
        return (
            torch.tensor(observations, device=self.device).float(),
            torch.tensor(actions, device=self.device).float(),
            torch.tensor(auxiliaries, device=self.device).float(),
            torch.tensor(values, device=self.device).float(),
        )
