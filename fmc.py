
import torch
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt


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
def _clone_states(dynamics_model, num_walkers: int, rewards, verbose: bool = False):
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

    return clone_partners, clone_mask


def _update_game_tree(game_tree: nx.DiGraph, walker_node_ids: torch.Tensor, actions, rewards, clone_partners, clone_mask, dynamics_model):
    num_walkers = dynamics_model.state.shape[0]

    candidate_walker_ids = torch.randint(99999, 999999999999, (num_walkers,))

    for walker_index in range(num_walkers):
        cloned = clone_mask[walker_index]

        if cloned:
            # TODO: prune tree (remove all nodes/edges before)
            continue

        previous_node = walker_node_ids[walker_index]
        new_node = candidate_walker_ids[walker_index]

        game_tree.add_node(new_node, state=dynamics_model.state[walker_index])
        game_tree.add_edge(previous_node, new_node, action=actions[walker_index], weight=rewards[walker_index])

    # clone new node ids
    walker_node_ids[clone_mask] = candidate_walker_ids[clone_partners[clone_mask]]


@torch.no_grad()
def lookahead(initial_state, dynamics_model, k: int, num_walkers: int = 4):
    # TODO: ensure the dynamics model weights remain fixed and have no gradients.

    action_history = []
    reward_history = []

    game_tree = nx.DiGraph()

    root = 0
    game_tree.add_node(root, state=initial_state)
    
    walker_node_ids = torch.zeros(num_walkers, dtype=torch.long)

    state = torch.zeros((num_walkers, *initial_state.shape))
    state[:] = initial_state

    dynamics_model.set_state(state)

    for _ in range(k):
        actions = _get_random_actions(dynamics_model, num_walkers)  # TODO: introduce a policy function
        rewards = dynamics_model.forward(actions)

        clone_partners, clone_mask = _clone_states(dynamics_model, num_walkers, rewards)

        _update_game_tree(game_tree, walker_node_ids, actions, rewards, clone_partners, clone_mask, dynamics_model)

    nx.draw(game_tree)
    plt.show()

    # TODO: select the best action
    return action_history[0][0, 0].numpy()



class FMC:
    def __init__(self, num_walkers: int, dynamics_model, initial_state, balance: float = 1, verbose: bool = False):
        self.num_walkers = num_walkers
        self.balance = balance
        self.verbose = verbose

        self.root = 0
        self.game_tree = nx.DiGraph()
        self.game_tree.add_node(self.root, state=initial_state)

        self.state = torch.zeros((num_walkers, *initial_state.shape))
        self.state[:] = initial_state

        self.dynamics_model = dynamics_model
        self.dynamics_model.set_state(self.state)

    @torch.no_grad()
    def _perturbate(self):
        actions = self._get_actions()
        self.rewards = self.dynamics_model.forward(actions)

    @torch.no_grad()
    def simulate(self, k: int):

        for _ in range(k):
            self._perturbate()
            self._clone_states()

            # _update_game_tree(game_tree, walker_node_ids, actions, rewards, clone_partners, clone_mask, dynamics_model)

        # sanity check
        assert self.dynamics_model.state.shape == (self.num_walkers, self.dynamics_model.embedding_size)

        return self._get_actions()[0, 0].numpy()  # TODO!

    @torch.no_grad()
    def _get_actions(self):
        # TODO: policy function
        
        actions = []
        for _ in range(self.num_walkers):
            action = self.dynamics_model.action_space.sample()
            actions.append(action)
        return torch.tensor(actions).unsqueeze(-1)

    @torch.no_grad()
    def _assign_clone_partners(self):
        self.clone_partners = np.random.choice(np.arange(self.num_walkers), size=self.num_walkers)

    @torch.no_grad()
    def _calculate_distances(self):
        self.distances = torch.linalg.norm(self.state - self.state[self.clone_partners], dim=1)

    @torch.no_grad()
    def _calculate_virtual_rewards(self):
        activated_rewards = _relativize_vector(self.rewards.squeeze(-1))
        activated_distances = _relativize_vector(self.distances)

        self.virtual_rewards = activated_rewards * activated_distances ** self.balance

    @torch.no_grad()
    def _determine_clone_mask(self):
        vr = self.virtual_rewards
        pair_vr = vr[self.clone_partners]

        self.clone_probabilities = (pair_vr - vr) / torch.where(vr > 0, vr, 1e-8)
        r = np.random.uniform()
        self.clone_mask = self.clone_probabilities >= r

    @torch.no_grad()
    def _clone_states(self):

        # prepare virtual rewards and partner virtual rewards
        self._assign_clone_partners()
        self._calculate_distances()
        self._calculate_virtual_rewards()
        self._determine_clone_mask()


        # TODO: don't clone best walker
        if self.verbose:
            print()
            print()
            print('clone stats:')
            print("state order", self.clone_partners)
            print("distances", self.distances)
            print("virtual rewards", self.virtual_rewards)
            print("clone probabilities", self.clone_probabilities)
            print("clone mask", self.clone_mask)
            print("state before", self.dynamics_model.state)

        # execute clone
        self.dynamics_model.state[self.clone_mask] = self.dynamics_model.state[self.clone_partners[self.clone_mask]]

        if self.verbose:
            print("state after", self.dynamics_model.state)