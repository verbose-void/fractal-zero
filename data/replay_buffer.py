import numpy as np


class GameHistory:

    def __init__(self, initial_observation):
        self.actions = [0]  # TODO: use the action shape
        self.observations = [initial_observation]
        self.environment_reward_signals = [0]
        self.values = [0]

    def append(self, action, observation, environment_reward_signal, value):
        self.actions.append(action)
        self.observations.append(observation)
        self.environment_reward_signals.append(environment_reward_signal)
        self.values.append(value)

    def __getitem__(self, index: int):
        return (self.observations[index], self.actions[index], self.environment_reward_signals[index], self.values[index])

    def __len__(self):
        if len(self.observations) == len(self.actions) == len(self.environment_reward_signals):
            return len(self.observations)
        raise ValueError(str(self))

    def __str__(self):
        return f"GameHistory(num_actions={len(self.actions)}, num_observation={len(self.observations)}, num_rewards={len(self.environment_reward_signals)})"


class ReplayBuffer:
    def __init__(self, max_size: int):
        # TODO: prioritized experience replay (PER) https://arxiv.org/abs/1511.05952

        self.max_size = max_size

        self.game_histories = []

    def append(self, game_history: GameHistory):
        if len(self) >= self.max_size:
            # first in, first out.
            self.game_histories.pop(0)

        self.game_histories.append(game_history)

        if len(self) > self.max_size:
            raise ValueError

    def sample(self) -> tuple:
        game_index = np.random.randint(0, len(self.game_histories))
        game_history = self.game_histories[game_index]
        frame_index = np.random.randint(0, len(game_history))
        return game_history[frame_index]

    def __len__(self):
        return len(self.game_histories)
