


class LatentEnv:
    def __init__(self, observation_space, action_space, model):
        self.observation_space = observation_space
        self.action_space = action_space
        self.model = model

    def step(self, action):
        pass