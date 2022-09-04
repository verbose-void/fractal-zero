import gym

import torch
from fmc import FMC

from models.dynamics import FullyConnectedDynamicsModel
from models.representation import FullyConnectedRepresentationModel
    

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    obs = env.reset()

    embedding_size = 16
    out_features = 1

    representation_model =  FullyConnectedRepresentationModel(env, embedding_size)
    dynamics_model = FullyConnectedDynamicsModel(env, embedding_size, out_features=out_features)

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
        print("reward", reward)
        if done:
            print('done')
            break

        # fmc.render_best_walker_path()
