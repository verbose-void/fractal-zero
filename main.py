import gym

from latent_env import LatentEnv

import torch

from models.dynamics import FullyConnectedDynamicsModel
from models.representation import FullyConnectedRepresentationModel



def lookahead(state, latent_env: LatentEnv, n: int):
    actions = []

    latent_env.set_state(state)

    for _ in range(n):
        action = torch.tensor(latent_env.action_space.sample())
        print(action.shape)

        latent_env.step(action)

        actions.append(action)

    return actions[0].numpy()
    

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    obs = env.reset()

    embedding_size = 16
    out_features = 2

    representation_model =  FullyConnectedRepresentationModel(env, embedding_size)
    dynamics_model = FullyConnectedDynamicsModel(env, embedding_size, out_features=out_features)

    latent_env = LatentEnv(env, model=dynamics_model)


    lookahead_steps = 10
    steps = 10
    for _ in range(steps):
        obs = torch.tensor(obs)

        state = representation_model.forward(obs)
        action = lookahead(state, latent_env, lookahead_steps)

        obs, reward, done, info = env.step(action)
        print("reward", reward)
        if done:
            print('done')
            break
