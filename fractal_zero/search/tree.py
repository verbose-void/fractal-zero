from copy import copy
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Sequence
from uuid import UUID, uuid4
import numpy as np


class StateNode:
    def __init__(
        self, observation, reward, num_child_walkers: int = 1, terminal: bool = False
    ):
        self.id = uuid4()
        self.observation = observation
        self.reward = reward
        self.terminal = terminal

        self.num_child_walkers = num_child_walkers
        self.visits = 1

    def __str__(self) -> str:
        return f"State(nc={self.num_child_walkers}, c={self.visits}, r={self.reward})"


class Path:
    ordered_states: List[StateNode]

    def __init__(self, root: StateNode, g: nx.Graph):
        self.root = root
        self.g = g
        self.ordered_states = [self.root]

    def add_node(self, node: StateNode):
        self.ordered_states.append(node)

    def clone_to(self, new_path: "Path", return_new_path: bool = True):
        if self.root != new_path.root:
            raise ValueError("Cannot clone to a path unless they share the same root.")

        common_state = None

        # backpropagate
        for state in reversed(self.ordered_states):
            # will break at root or the closest common state
            if state in new_path.ordered_states:
                common_state = state
                break

            # only decrement BEFORE the common state.
            state.num_child_walkers -= 1

        # TODO: maybe don't clear, don't loop through all new path ordered states, begin at the common node.
        cloning_path = self
        if return_new_path:
            cloning_path = Path(self.root, self.g)

        # don't copy, just update reference
        saw_common = False
        cloning_path.ordered_states.clear()  
        last_state = None
        for new_state in new_path.ordered_states:
            if new_state is common_state:
                saw_common = True
            elif saw_common:
                # only increment AFTER the common state.
                new_state.num_child_walkers += 1

            # sanity check
            if last_state and not cloning_path.g.has_edge(last_state, new_state):
                raise ValueError(f"No edge exists between {last_state} and {new_state}.")
                
            new_state.visits += 1
            cloning_path.add_node(new_state)
            last_state = new_state

        if return_new_path:
            return cloning_path

    def prune(self):
        # pruning should only occur on paths that are going to be discarded from the tree.
        for state in reversed(self.ordered_states):
            should_prune = state.num_child_walkers <= 0 and self.g.has_node(state)
            if should_prune:
                self.g.remove_node(state)
            else:
                # if any states have > 0 num child walkers, all of their parents should as well.
                # also, if a state was already pruned, it's safe to assume their parents were as well,
                # so we can break.
                break

    @property
    def total_reward(self) -> float:
        return float(sum([s.reward for s in self.ordered_states]))

    @property
    def average_reward(self) -> float:
        return self.total_reward / len(self)

    @property
    def last_node(self):
        return self.ordered_states[-1]

    @property
    def last_action(self):
        if len(self.ordered_states) < 2:
            return None
        u, v = self.ordered_states[-2], self.ordered_states[-1]
        return self.get_action_between(u, v)

    def get_action_between(self, state, next_state):
        edge_data = self.g.get_edge_data(state, next_state)
        return edge_data["action"]

    def __len__(self):
        return len(self.ordered_states)

    def __iter__(self):
        self._iter = 0
        return self

    def __next__(self):
        # stop one early because the last state should have no child nodes
        # therefore there won't be any actions registered for that state for this path.
        if self._iter >= len(self) - 1:
            raise StopIteration

        state = self.ordered_states[self._iter]
        next_state = self.ordered_states[self._iter + 1]

        action = self.get_action_between(state, next_state)

        self._iter += 1
        return state, action


class GameTree:
    def __init__(self, num_walkers: int, root_observation=None, prune: bool = True):
        self.num_walkers = num_walkers
        self.prune = prune

        num_children = self.num_walkers
        self.root = StateNode(
            root_observation,
            reward=0,
            num_child_walkers=num_children,
            terminal=False,
        )

        self.g = nx.DiGraph()
        self.g.add_node(self.root)

        self.walker_paths = [Path(self.root, self.g) for _ in range(self.num_walkers)]

    def build_next_level(
        self,
        actions: Sequence,
        new_observations: Sequence,
        rewards: Sequence,
        freeze_mask=None,
    ):
        if freeze_mask is None:
            freeze_mask = np.zeros(self.num_walkers, dtype=bool)

        assert (
            len(actions)
            == len(new_observations)
            == len(rewards)
            == len(freeze_mask)
            == self.num_walkers
        )

        # TODO: how can we detect duplicate observations / action transitions to save memory? (might not be super important)
        it = zip(self.walker_paths, actions, new_observations, rewards, freeze_mask)
        for path, action, new_observation, reward, frozen in it:
            if frozen:  
                continue

            last_node = path.last_node

            # TODO: denote terminal states
            new_node = StateNode(new_observation, reward, terminal=False)
            path.add_node(new_node)

            self.g.add_edge(last_node, new_node, action=action)

    def clone(self, partners: Sequence, clone_mask: Sequence):
        new_paths = []
        old_paths = []

        for i, path in enumerate(self.walker_paths):

            if not clone_mask[i]:
                # do not clone
                new_paths.append(path)
                continue

            target_path = self.walker_paths[partners[i]]
            new_path = path.clone_to(target_path, return_new_path=True)
            new_paths.append(new_path)
            old_paths.append(path)

        if self.prune:
            for path in old_paths:
                path.prune()

        self.walker_paths = new_paths

    @property
    def best_path(self):
        return max(
            self.walker_paths, key=lambda p: p.total_reward
        )  # best path of current walker

    @property
    def last_actions(self):
        return [p.last_action for p in self.walker_paths]

    def get_depths(self) -> np.ndarray:
        depths = np.zeros(self.num_walkers, dtype=float)
        for i, path in enumerate(self.walker_paths):
            depths[i] = len(path)
        return depths

    def render(self):
        colors = []
        for node in self.g.nodes:
            if node == self.root:
                colors.append("green")
            else:
                colors.append("red")

        nx.draw(self.g, with_labels=True, node_color=colors, node_size=80)
        plt.show()
