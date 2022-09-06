
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
    """Fractal Monte Carlo is a collaborative cellular automata based tree search algorithm. This version is special, because instead of having a gym
    environment maintain the state for each walker during the search process, each walker's state is represented inside of a batched hidden
    state variable inside of a dynamics model. Basically, the dynamics model's hidden state is of shape (num_walkers, *embedding_shape).

    This is inspired by Muzero's technique to have a dynamics model be learned such that the tree search need not interact with the environment
    itself. With FMC, it is much more natural than with MCTS, mostly because of the cloning phase being contrastive. As an added benefit of this
    approach, it's natively vectorized so it can be put onto the GPU.
    """

    def __init__(self, num_walkers: int, dynamics_model, initial_state, balance: float = 1, verbose: bool = False):
        self.num_walkers = num_walkers
        self.balance = balance
        self.verbose = verbose

        self.state = torch.zeros((num_walkers, *initial_state.shape))
        self.state[:] = initial_state

        # self.root = 0
        # self.game_tree = nx.DiGraph()
        # self.game_tree.add_node(self.root, state=initial_state)
        # self.walker_node_ids = torch.zeros(self.num_walkers, dtype=torch.long)

        self.dynamics_model = dynamics_model
        self.dynamics_model.set_state(self.state)

    @torch.no_grad()
    def _perturbate(self):
        """Advance the state of each walker."""

        self._assign_actions()
        self.rewards = self.dynamics_model.forward(self.actions)

        self.reward_buffer[:, self.simulation_iteration] = self.rewards

    @torch.no_grad()
    def simulate(self, k: int):
        """Run FMC for k iterations, returning the best action that was taken at the root/initial state."""

        assert k > 0

        self.reward_buffer = torch.zeros(size=(self.num_walkers, k, 1))
        self.root_actions = None

        for self.simulation_iteration in range(k):
            self._perturbate()
            self._prepare_clone_variables()

            # TODO: explain backpropagate all
            backpropagate_all = self.simulation_iteration == k-1
            # self._backpropagate_reward_buffer(backpropagate_all=backpropagate_all)

            self._execute_cloning()

            # self._update_game_tree()

            # nx.draw(self.game_tree)
            # plt.show()

        # sanity check
        assert self.dynamics_model.state.shape == (self.num_walkers, self.dynamics_model.embedding_size)

        # TODO: try to convert the root action distribution into a policy distribution? this may get hard in continuous action spaces. https://arxiv.org/pdf/1805.09613.pdf
        
        # TODO: backpropagate reward to estimate a value function?
        # NOTE: for each FMC simulation, we only need to store the value of the root. so the entire simulation collapses into a single point
        # to determine the root node's approximate value.

        # return self._pick_root_action()
        return self.actions[0, 0].numpy()  # TODO

    @torch.no_grad()
    def _assign_actions(self):
        """Each walker picks an action to advance it's state."""

        # TODO: policy function?
        actions = []
        for _ in range(self.num_walkers):
            action = self.dynamics_model.action_space.sample()
            actions.append(action)
        self.actions = torch.tensor(actions).unsqueeze(-1)

        if self.root_actions is None:
            self.root_actions = torch.tensor(self.actions)

    @torch.no_grad()
    def _assign_clone_partners(self):
        """For the cloning phase, walkers need a partner to determine if they should be sent as reinforcements to their partner's state."""
            
        self.clone_partners = np.random.choice(np.arange(self.num_walkers), size=self.num_walkers)

    @torch.no_grad()
    def _calculate_distances(self):
        """For the cloning phase, we calculate the distances between each walker and their partner for balancing exploration."""

        self.distances = torch.linalg.norm(self.state - self.state[self.clone_partners], dim=1)

    @torch.no_grad()
    def _calculate_virtual_rewards(self):
        """For the cloning phase, we calculate a virtual reward that is the composite of each walker's distance to their partner weighted with
        their rewards. This is used to determine the probability to clone and is used to balance exploration and exploitation.

        Both the reward and distance vectors are "relativized". This keeps all of the values in each vector contextually scaled with each step.
        The authors of Fractal Monte Carlo claim this is a method of shaping a "universal reward function". Without relativization, the
        vectors may have drastically different ranges, causing more volatility in how many walkers are cloned at each step. If the reward or distance
        ranges were too high, it's likely no cloning would occur at all. If either were too small, then it's likely all walkers would be cloned.
        """

        activated_rewards = _relativize_vector(self.rewards.squeeze(-1))
        activated_distances = _relativize_vector(self.distances)

        self.virtual_rewards = activated_rewards * activated_distances ** self.balance

    @torch.no_grad()
    def _determine_clone_mask(self):
        """The clone mask is based on the virtual rewards of each walker and their clone partner. If a walker is selected to clone, their
        state will be replaced with their partner's state.
        """

        vr = self.virtual_rewards
        pair_vr = vr[self.clone_partners]

        self.clone_probabilities = (pair_vr - vr) / torch.where(vr > 0, vr, 1e-8)
        r = np.random.uniform()
        self.clone_mask = self.clone_probabilities >= r

    @torch.no_grad()
    def _prepare_clone_variables(self):
        # TODO: docstring

        # prepare virtual rewards and partner virtual rewards
        self._assign_clone_partners()
        self._calculate_distances()
        self._calculate_virtual_rewards()
        self._determine_clone_mask()

    @torch.no_grad()
    def _execute_cloning(self):
        """The cloning phase is where the collaboration of the cellular automata comes from. Using the virtual rewards calculated for
        each walker and clone partners that are randomly assigned, there is a probability that some walkers will be sent as reinforcements
        to their randomly assigned clone partner.

        The goal of the clone phase is to maintain a balanced density over state occupations with respect to exploration and exploitation.   
        """

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
        self.actions[self.clone_mask] = self.actions[self.clone_partners[self.clone_mask]]
        self.reward_buffer[self.clone_mask] = self.reward_buffer[self.clone_partners[self.clone_mask]]
        self.root_actions[self.clone_mask] = self.root_actions[self.clone_partners[self.clone_mask]]

        if self.verbose:
            print("state after", self.dynamics_model.state)

    def _backpropagate_reward_buffer(self, backpropagate_all: bool = False):

        # TODO: only "backpropagate" reward histories for the walkers who's clone mask value is True.
        pass

    # @torch.no_grad()
    # def _update_game_tree(self):
    #     """A tree of walker trajectories is maintained so that after the simulation is complete, we can analyze the actions
    #     taken and choose the best trajectory according to the reward estimations.
    #     """

    #     candidate_walker_ids = torch.randint(1, 9999999, (self.num_walkers,))
    #     candidate_walker_ids[self.clone_mask] = candidate_walker_ids[self.clone_partners[self.clone_mask]]

    #     nodes_to_backpropagate = set()

    #     for walker_index in range(self.num_walkers):
    #         previous_node = self.walker_node_ids[walker_index].item()

    #         # TODO: prune tree when there are no walkers in a path
    #         cloned = self.clone_mask[walker_index]
    #         if cloned:
    #             nodes_to_backpropagate.add(previous_node)
    #             continue

    #         new_node = candidate_walker_ids[walker_index].item()

    #         state = self.dynamics_model.state[walker_index]
    #         reward = self.rewards[walker_index].item()
    #         # self.game_tree.add_node(new_node, state=state, reward=reward)

    #         # path_weight = 9999999 - reward
    #         # self.game_tree.add_edge(previous_node, new_node, action=self.actions[walker_index], path_weight=path_weight)

    #     for node in nodes_to_backpropagate:
    #         self.backpropagate(node)

    #     self.walker_node_ids[:] = candidate_walker_ids

    # def _determine_best_walker_path(self):
    #     """The best walker path is based on the path through the game tree, beginning at the root and ending at *one* of the current walker's 
    #     state nodes, that results in the highest sum of edge weights. In other words, it's the set of actions taken that resulted in the highest
    #     cumulative reward.
    #     """

    #     # TODO optimize this!
    #     # TODO: we might not need to store the entire tree, but rather just store the action that was taken at the root for each walker.

    #     best_path = None
    #     lowest_distance = float("inf")

    #     for walker_index in range(self.num_walkers):
    #         walker_node = self.walker_node_ids[walker_index].item()

    #         # find longest path between root and current walker node
    #         distance, path = nx.single_source_dijkstra(self.game_tree, self.root, walker_node, weight="path_weight")

    #         if distance < lowest_distance:
    #             lowest_distance = distance
    #             best_path = path

    #     if len(best_path) <= 1:
    #         raise ValueError(f"The best path length must be >= 2. Got: {best_path}.")

    #     self.best_path = best_path

    # @torch.no_grad()
    # def _pick_root_action(self):
    #     """The root action is chosen based on the best path/trajectory taken by a walker in the game tree."""

    #     self._determine_best_walker_path()
    #     u, v = self.best_path[0], self.best_path[1]

    #     # sanity check
    #     if u != self.root:
    #         raise ValueError(f"The first node in the selected path is expected to be the root node ({self.root}), instead got: {u}.")

    #     action = self.game_tree.edges[u, v]["action"]
    #     return action[0].numpy()

    def render_best_walker_path(self):
        edges = [(self.best_path[i], self.best_path[i+1]) for i in range(len(self.best_path) - 1)]
        color_map = ['green' if node == self.root else 'black' for node in self.game_tree]
        edge_color = ['red' if edge in edges else "black" for edge in self.game_tree.edges]
        nx.draw(self.game_tree, node_color=color_map, edge_color=edge_color)
        plt.show()