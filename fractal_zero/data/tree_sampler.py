import torch

from fractal_zero.search.tree import GameTree


class TreeSampler:
    def __init__(self, tree: GameTree, sample_type: str="all_nodes"):
        self.tree = tree

        # TODO: parameter for weighting type (constant, walker_vists/clone_receives, walker_children, etc.)

        self.sample_type = sample_type

    def _get_best_path_as_batch(self):
        observations = []
        actions = []

        path = self.tree.best_path
        for state, action in path:
            # obs = torch.tensor(state.observation)
            observations.append(state.observation)
            actions.append([action])

        weights = torch.ones((1, 1, len(observations))).float()
        return observations, actions, weights

    def _get_all_nodes_batch(self):
        observations = []
        child_actions = []
        child_weights = []

        g = self.tree.g
        for node in g.nodes:
            observations.append(node.observation)

            actions = []
            weights = []
            for _, child_node, data in g.out_edges(node, data=True):
                weight = child_node.num_child_walkers / self.tree.num_walkers

                weights.append(weight)

                action = data["action"]
                actions.append(action)

            child_actions.append(actions)
            child_weights.append(torch.tensor(weights).float())

        return observations, child_actions, child_weights
        
    def get_batch(self):
        if self.sample_type == "best_path":
            obs, acts, weights = self._get_best_path_as_batch()
        elif self.sample_type == "all_nodes":
            obs, acts, weights = self._get_all_nodes_batch()
        else:
            raise ValueError(f"Sample type {self.sample_type} is not supported.")

        # sanity check
        if not (len(obs) == len(acts) == len(weights)):
            raise ValueError(f"Got different lengths for batch return: {len(obs)}, {len(acts)}, {len(weights)}.")

        return obs, acts, weights
        