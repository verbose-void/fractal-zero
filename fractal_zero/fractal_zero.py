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

        _, value_estimate = self.model.prediction_model.forward(state)

        if self.config.lookahead_steps > 0:
            self.fmc.set_state(state)
            action = self.fmc.simulate(k, greedy_action=greedy_action)
            return action, self.fmc.root_value, value_estimate.item()

        raise NotImplementedError("Action prediction not yet working.")

    def play_game(
        self,
        render: bool = False,
    ):
        env = self.config.env

        obs = env.reset()
        game_history = GameHistory(obs)

        self.fmc = FMC(self.config, verbose=False)

        for step in range(self.config.max_game_steps):
            obs = torch.tensor(obs, device=self.config.device)
            action, root_value, value_estimate = self.forward(obs)
            obs, reward, done, info = env.step(action)

            game_history.append(action, obs, reward, root_value)

            if render:
                print()
                print(f"step={step}")
                print(f"reward={reward}, done={done}, info={info}")
                print(
                    f"action={action}, root_value={root_value}, value_estimate={value_estimate}"
                )
                env.render()
                sleep(0.1)

            if done:
                break

        if render:
            print()
            print("game summary:")
            print(f"cumulative rewards: {sum(game_history.environment_reward_signals)}")
            print(f"episode length: {len(game_history)}")

        return game_history
