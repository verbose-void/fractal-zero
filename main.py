import gym

import torch
from fmc import FMC

from models.dynamics import FullyConnectedDynamicsModel
from models.joint_model import JointModel
from models.representation import FullyConnectedRepresentationModel
from replay_buffer import GameHistory, ReplayBuffer
from trainer import Trainer


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

        # env.render()
        # fmc.render_best_walker_path()

    return game_history
    

if __name__ == "__main__":
    env = gym.make('CartPole-v0')

    embedding_size = 16
    out_features = 1

    representation_model =  FullyConnectedRepresentationModel(env, embedding_size)
    dynamics_model = FullyConnectedDynamicsModel(env, embedding_size, out_features=out_features)
    joint_model = JointModel(representation_model, dynamics_model)

    replay_buffer = ReplayBuffer()
    trainer = Trainer(replay_buffer, joint_model)

    num_games = 3
    for _ in range(num_games):
        game_history = play_game(env, representation_model, dynamics_model)
        print(game_history)
        replay_buffer.append(game_history)
        

        
