from time import sleep

import torch
from fractal_zero.config import FractalZeroConfig

from fractal_zero.data.replay_buffer import GameHistory
from fractal_zero.fmc import FMC
from fractal_zero.models.joint_model import JointModel


class FractalZero(torch.nn.Module):
    def __init__(self, config: FractalZeroConfig):
        super().__init__()

        self.config = config

        self.model = self.config.joint_model

        # TODO: reuse FMC instance?
        self.fmc = None

    def forward(self, observation):
        # TODO: docstring, note that lookahead_steps == 0 means there won't be a tree search

        state = self.model.representation_model.forward(observation)

        if self.training:
            greedy_action = False
            k = self.config.lookahead_steps
        else:
            greedy_action = True
            k = self.config.evaluation_lookahead_steps

        if self.config.lookahead_steps > 0:
            self.fmc.set_state(state)
            action = self.fmc.simulate(k, greedy_action=greedy_action)
            return action, self.fmc.root_value

        raise NotImplementedError("Action prediction not yet working.")
        action, value_estimate = self.model.prediction_model.forward(state)
        return action, value_estimate

    def play_game(
        self,
        render: bool = False,
    ):
        # TODO: create config class

        env = self.config.env

        obs = env.reset()
        game_history = GameHistory(obs)

        self.fmc = FMC(self.config, verbose=False)

        for _ in range(self.config.max_game_steps):
            obs = torch.tensor(obs, device=self.config.device)
            action, value_estimate = self.forward(obs)
            obs, reward, done, info = env.step(action)

            game_history.append(action, obs, reward, value_estimate)

            if done:
                break

            if render:
                env.render()
                sleep(0.1)

        return game_history
