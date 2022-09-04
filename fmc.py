
import torch
import numpy as np


def get_random_actions(dynamics_model, n: int):
    actions = []
    for _ in range(n):
        action = dynamics_model.action_space.sample()
        actions.append(action)
    return torch.tensor(actions).unsqueeze(-1)


def get_random_state_order(num_walkers: int):
    return np.random.choice(np.arange(num_walkers), size=num_walkers)


def get_distances(states, state_order):
    return torch.linalg.norm(states - states[state_order], dim=1)
    
    
def _relativize_vector(vector):
    std = vector.std()
    if std == 0:
        return torch.ones(len(vector))
    standard = (vector - vector.mean()) / std
    standard[standard > 0] = torch.log(1 + standard[standard > 0]) + 1
    standard[standard <= 0] = torch.exp(standard[standard <= 0])
    return standard


def get_virtual_rewards(rewards, distances, balance: float = 1):
    # TODO: reletivize

    activated_rewards = _relativize_vector(rewards.squeeze(-1))
    activated_distances = _relativize_vector(distances)

    return activated_rewards * activated_distances ** balance


@torch.no_grad()
def do_state_cloning(dynamics_model, num_walkers: int, rewards):
    assert dynamics_model.state.shape == (num_walkers, dynamics_model.embedding_size)

    state_order = get_random_state_order(num_walkers)
    distances = get_distances(dynamics_model.state, state_order)
    vr = get_virtual_rewards(rewards, distances)

    # TODO: don't clone best walker
    print()
    print()
    print()
    print()
    print('clone stats:')
    print("state order", state_order)
    print("distances", distances)
    print("virtual rewards", vr)

    pair_vr = vr[state_order]

    clone_probabilities = (pair_vr - vr) / torch.where(vr > 0, vr, 1e-8)
    r = np.random.uniform()

    clone_mask = clone_probabilities >= r

    print("clone probabilities", clone_probabilities)
    print("clone mask", clone_mask)
    

    # print()
    print("state before", dynamics_model.state)

    dynamics_model.state[clone_mask] = dynamics_model.state[state_order[clone_mask]]
    print("state after", dynamics_model.state)


@torch.no_grad()
def lookahead(initial_state, dynamics_model, k: int, num_walkers: int = 4):
    # TODO: ensure the dynamics model weights remain fixed and have no gradients.

    action_history = []
    reward_history = []

    state = torch.zeros((num_walkers, *initial_state.shape))
    state[:] = initial_state

    dynamics_model.set_state(state)

    for _ in range(k):
        actions = get_random_actions(dynamics_model, num_walkers)
        print(state.shape, actions.shape)

        rewards = dynamics_model.forward(actions)

        action_history.append(actions)
        reward_history.append(rewards)

        do_state_cloning(dynamics_model, num_walkers, rewards)

        print(rewards.shape)


    # TODO: select the best action
    return action_history[0][0, 0].numpy()