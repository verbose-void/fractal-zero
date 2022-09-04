import gym

import torch

from models.dynamics import FullyConnectedDynamicsModel
from models.representation import FullyConnectedRepresentationModel


def get_random_actions(dynamics_model, n: int):
    actions = []
    for _ in range(n):
        action = dynamics_model.action_space.sample()
        actions.append(action)
    return torch.tensor(actions).unsqueeze(-1)


def lookahead(initial_state, dynamics_model, k: int, num_walkers: int = 4):
    action_history = []

    state = torch.zeros((num_walkers, *initial_state.shape))
    state[:] = initial_state

    dynamics_model.set_state(state)

    for _ in range(k):
        actions = get_random_actions(dynamics_model, num_walkers)
        print(state.shape, actions.shape)

        dynamics_model.forward(actions)
        action_history.append(actions)

    # TODO: select the best action
    return action_history[0][0, 0].numpy()
    

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
