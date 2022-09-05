

class GameHistory:

    def __init__(self, initial_observation):
        self.actions = [None]
        self.observations = [initial_observation]
        self.rewards = [None]

    def append(self, action, observation, reward):
        self.actions.append(action)
        self.observations.append(observation)
        self.rewards.append(reward)

    def __str__(self):
        return f"GameHistory(num_actions={len(self.actions)}, num_observation={len(self.observations)}, num_rewards={len(self.rewards)})"


class ReplayBuffer:
    def __init__(self):
        self.games = []

    def append(self, game_history: GameHistory):
        self.games.append(game_history)