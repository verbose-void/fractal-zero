import torch
import numpy as np

from fractal_zero.search.tree import GameTree, StateNode


class TreeSampler:
    def __init__(self, tree: GameTree, sample_type: str="all_nodes", weight_type: str="walker_children_ratio"):
        self.tree = tree
        self.sample_type = sample_type
        self.weight_type = weight_type

        if not self.tree.prune:
            raise NotImplementedError("TreeSampling on an unpruned tree has not been considered.")

    def _calculate_weight(self, node: StateNode) -> float:
        if self.weight_type == "walker_children_ratio":
            return node.num_child_walkers / self.tree.num_walkers
        elif self.weight_type == "constant":
            return 1.0
        elif self.weight_type == "time_spent_at_node":
            raise NotImplementedError
        raise ValueError(f"{self.weight_type} not supported.")

    def _get_best_path_as_batch(self):
        observations = []
        actions = []
        weights = []

        path = self.tree.best_path
        for state, action in path:
            # NOTE: never skip due to weight!
            weight = self._calculate_weight(state)
            weights.append([weight])

            observations.append(state.observation)
            actions.append([action])

        return observations, actions, weights

    def _get_all_nodes_batch(self):
        observations = []
        child_actions = []
        child_weights = []

        g = self.tree.g
        for node in g.nodes:

            actions = []
            weights = []
            for _, child_node, data in g.out_edges(node, data=True):
                weight = self._calculate_weight(child_node)

                # skip if the weight is almost 0.
                if np.isclose(weight, 0):
                    continue

                weights.append(weight)
                action = data["action"]
                actions.append(action)

            # if no action targets exist, skip this state.
            if len(actions) <= 0:
                continue

            observations.append(node.observation)
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
        