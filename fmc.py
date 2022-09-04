
import torch
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

    
@torch.no_grad()
def _relativize_vector(vector):
    std = vector.std()
    if std == 0:
        return torch.ones(len(vector))
    standard = (vector - vector.mean()) / std
    standard[standard > 0] = torch.log(1 + standard[standard > 0]) + 1
    standard[standard <= 0] = torch.exp(standard[standard <= 0])
    return standard

class FMC:
    def __init__(self, num_walkers: int, dynamics_model, initial_state, balance: float = 1, verbose: bool = False):
        self.num_walkers = num_walkers
        self.balance = balance
        self.verbose = verbose

        self.state = torch.zeros((num_walkers, *initial_state.shape))
        self.state[:] = initial_state

        self.root = 0
        self.game_tree = nx.DiGraph()
        self.game_tree.add_node(self.root, state=initial_state)
        self.walker_node_ids = torch.zeros(self.num_walkers, dtype=torch.long)

        self.dynamics_model = dynamics_model
        self.dynamics_model.set_state(self.state)

    @torch.no_grad()
    def _perturbate(self):
        self._assign_actions()
        self.rewards = self.dynamics_model.forward(self.actions)

    @torch.no_grad()
    def simulate(self, k: int):

        for _ in range(k):
            self._perturbate()
            self._clone_states()
            self._update_game_tree()

            # nx.draw(self.game_tree)
            # plt.show()

        # sanity check
        assert self.dynamics_model.state.shape == (self.num_walkers, self.dynamics_model.embedding_size)

        return self.actions[0, 0].numpy()  # TODO select the actual best action

    @torch.no_grad()
    def _assign_actions(self):
        # TODO: policy function
        
        actions = []
        for _ in range(self.num_walkers):
            action = self.dynamics_model.action_space.sample()
            actions.append(action)
        self.actions = torch.tensor(actions).unsqueeze(-1)

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

    @torch.no_grad()
    def _update_game_tree(self):
        candidate_walker_ids = torch.randint(1, 9999999, (self.num_walkers,))

        for walker_index in range(self.num_walkers):
            cloned = self.clone_mask[walker_index]

            if cloned:
                # TODO: prune tree (remove all nodes/edges before)
                continue

            previous_node = self.walker_node_ids[walker_index].item()
            new_node = candidate_walker_ids[walker_index].item()

            self.game_tree.add_node(new_node, state=self.dynamics_model.state[walker_index])
            self.game_tree.add_edge(previous_node, new_node, action=self.actions[walker_index], weight=self.rewards[walker_index])

        candidate_walker_ids[self.clone_mask] = candidate_walker_ids[self.clone_partners[self.clone_mask]]
        self.walker_node_ids[:] = candidate_walker_ids

        # clone new node ids
        # self.walker_node_ids[:] = candidate_walker_ids
        # self.walker_node_ids[self.clone_mask] = candidate_walker_ids[self.clone_partners[self.clone_mask]]