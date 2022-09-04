import gym

import torch
from fmc import FMC, lookahead

from models.dynamics import FullyConnectedDynamicsModel
from models.representation import FullyConnectedRepresentationModel
    

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    obs = env.reset()

    embedding_size = 16
    out_features = 1

    representation_model =  FullyConnectedRepresentationModel(env, embedding_size)
    dynamics_model = FullyConnectedDynamicsModel(env, embedding_size, out_features=out_features)


    lookahead_steps = 10
    steps = 10
    for _ in range(steps):
        obs = torch.tensor(obs)

        state = representation_model.forward(obs)

        # action = lookahead(state, dynamics_model, lookahead_steps)
        action = FMC(4, dynamics_model, state).simulate(lookahead_steps)

        obs, reward, done, info = env.step(action)
        print("reward", reward)
        if done:
            print('done')
            break
