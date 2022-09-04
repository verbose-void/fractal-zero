
import torch
import numpy as np


@torch.no_grad()
def _get_random_actions(dynamics_model, n: int):
    actions = []
    for _ in range(n):
        action = dynamics_model.action_space.sample()
        actions.append(action)
    return torch.tensor(actions).unsqueeze(-1)


@torch.no_grad()
def _get_clone_partners(num_walkers: int):
    return np.random.choice(np.arange(num_walkers), size=num_walkers)


@torch.no_grad()
def _get_distances(states, clone_partners):
    return torch.linalg.norm(states - states[clone_partners], dim=1)
    
    
@torch.no_grad()
def _relativize_vector(vector):
    std = vector.std()
    if std == 0:
        return torch.ones(len(vector))
    standard = (vector - vector.mean()) / std
    standard[standard > 0] = torch.log(1 + standard[standard > 0]) + 1
    standard[standard <= 0] = torch.exp(standard[standard <= 0])
    return standard


@torch.no_grad()
def _get_virtual_rewards(rewards, distances, balance: float = 1):
    # TODO: reletivize

    activated_rewards = _relativize_vector(rewards.squeeze(-1))
    activated_distances = _relativize_vector(distances)

    return activated_rewards * activated_distances ** balance


@torch.no_grad()
def _do_state_cloning(dynamics_model, num_walkers: int, rewards, verbose: bool = False):
    assert dynamics_model.state.shape == (num_walkers, dynamics_model.embedding_size)

    # prepare virtual rewards and partner virtual rewards
    clone_partners = _get_clone_partners(num_walkers)
    distances = _get_distances(dynamics_model.state, clone_partners)
    vr = _get_virtual_rewards(rewards, distances)
    pair_vr = vr[clone_partners]

    # prepare clone mask
    clone_probabilities = (pair_vr - vr) / torch.where(vr > 0, vr, 1e-8)
    r = np.random.uniform()
    clone_mask = clone_probabilities >= r

    # TODO: don't clone best walker
    if verbose:
        print()
        print()
        print('clone stats:')
        print("state order", clone_partners)
        print("distances", distances)
        print("virtual rewards", vr)
        print("clone probabilities", clone_probabilities)
        print("clone mask", clone_mask)
        print("state before", dynamics_model.state)

    # execute clone
    dynamics_model.state[clone_mask] = dynamics_model.state[clone_partners[clone_mask]]

    if verbose:
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
        actions = _get_random_actions(dynamics_model, num_walkers)
        rewards = dynamics_model.forward(actions)

        action_history.append(actions)
        reward_history.append(rewards)

        _do_state_cloning(dynamics_model, num_walkers, rewards)

    # TODO: select the best action
    return action_history[0][0, 0].numpy()