import gym

import torch

from models.dynamics import FullyConnectedDynamicsModel
from models.representation import FullyConnectedRepresentationModel



def lookahead(state, dynamics_model, k: int):
    actions = []

    dynamics_model.set_state(state)

    for _ in range(k):
        action = torch.tensor(dynamics_model.action_space.sample())
        dynamics_model.forward(action)
        actions.append(action)

    return actions[0].numpy()
    

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    obs = env.reset()

    embedding_size = 16
    out_features = 2

    representation_model =  FullyConnectedRepresentationModel(env, embedding_size)
    dynamics_model = FullyConnectedDynamicsModel(env, embedding_size, out_features=out_features)


    lookahead_steps = 10
    steps = 10
    for _ in range(steps):
        obs = torch.tensor(obs)

        state = representation_model.forward(obs)
        action = lookahead(state, dynamics_model, lookahead_steps)

        obs, reward, done, info = env.step(action)
        print("reward", reward)
        if done:
            print('done')
            break
