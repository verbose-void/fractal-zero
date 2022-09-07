from time import sleep
import gym
from data.replay_buffer import GameHistory
from fmc import FMC

import torch

from models.joint_model import JointModel



class FractalZero:
    def __init__(self, env: gym.Env, model: JointModel):
        self.env = env
        self.model = model

    def play_game(self, max_steps: int, num_walkers: int, lookahead_steps: int, render: bool = False) -> GameHistory:
        # TODO: create config class

        obs = self.env.reset()
        game_history = GameHistory(obs)

        fmc = FMC(num_walkers, self.model)

        for _ in range(max_steps):
            obs = torch.tensor(obs, device=self.model.device)

            # TODO: move this into FMC
            state = self.model.representation_model.forward(obs)
            fmc.set_state(state)

            action = fmc.simulate(lookahead_steps)
            obs, reward, done, info = self.env.step(action)

            game_history.append(action, obs, reward, fmc.root_value)

            if done:
                break

            if render:
                self.env.render()
                sleep(0.1)

        return game_history