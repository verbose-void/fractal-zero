

class GameHistory:

    def __init__(self, initial_observation):
        self.actions = [None]
        self.observations = [initial_observation]
        self.environment_reward_signals = [None]

    def append(self, action, observation, environment_reward_signal):
        self.actions.append(action)
        self.observations.append(observation)
        self.environment_reward_signals.append(environment_reward_signal)

    def __str__(self):
        return f"GameHistory(num_actions={len(self.actions)}, num_observation={len(self.observations)}, num_rewards={len(self.environment_reward_signals)})"


class ReplayBuffer:
    def __init__(self):
        self.game_histories = []

    def append(self, game_history: GameHistory):
        self.game_histories.append(game_history)