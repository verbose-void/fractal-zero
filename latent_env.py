import torch
import gym


class LatentEnv:
    def __init__(self, env: gym.Env, model: torch.nn.Module):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.model = model

    def set_state(self, state):
        self.model.set_state(state)

    def step(self, action):
        if len(action.shape) == 0:
            action = torch.unsqueeze(action, 0)

        reward = self.model.forward(action)
        return reward