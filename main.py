import gym

import torch
from fmc import FMC

from models.dynamics import FullyConnectedDynamicsModel
from models.representation import FullyConnectedRepresentationModel


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


def play_game(env, representation_model, dynamics_model) -> GameHistory:
    obs = env.reset()
    game_history = GameHistory(obs)

    num_walkers = 4
    lookahead_steps = 10
    steps = 10
    for _ in range(steps):
        obs = torch.tensor(obs)

        state = representation_model.forward(obs)

        # action = lookahead(state, dynamics_model, lookahead_steps)
        fmc = FMC(num_walkers, dynamics_model, state)
        action = fmc.simulate(lookahead_steps)

        obs, reward, done, info = env.step(action)

        game_history.append(action, obs, reward)

        print("reward", reward)
        if done:
            print('done')
            break

        # fmc.render_best_walker_path()

    return game_history
    

if __name__ == "__main__":
    env = gym.make('CartPole-v0')

    embedding_size = 16
    out_features = 1

    representation_model =  FullyConnectedRepresentationModel(env, embedding_size)
    dynamics_model = FullyConnectedDynamicsModel(env, embedding_size, out_features=out_features)

    game_history = play_game(env, representation_model, dynamics_model)
    print(game_history)

        
