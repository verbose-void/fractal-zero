import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Sequence
from uuid import UUID, uuid4


class StateNode:
    def __init__(self, observation, reward, num_child_walkers: int = 1, terminal: bool = False):
        self.id = uuid4()
        self.observation = observation
        self.reward = reward
        self.terminal = terminal
        
        self.num_child_walkers = num_child_walkers

    def __str__(self) -> str:
        return f"{self.num_child_walkers}"

    def __repr__(self):
        return self.__str__


class Path:
    ordered_states: List[StateNode]

    def __init__(self, root: StateNode, g: nx.Graph):
        self.root = root
        self.ordered_states = [root]
        self.g = g

    def add_node(self, node: StateNode):
        self.ordered_states.append(node)

    def clone_to(self, new_path: "Path", prune: bool = True):
        if self.root != new_path.root:
            raise ValueError("Cannot clone to a path unless they share the same root.")

        # build iterator
        reversed_indices = reversed(range(len(self.ordered_states)))
        reversed_states = reversed(self.ordered_states)
        reversed_new_states = reversed(new_path.ordered_states)
        it = zip(reversed_indices, reversed_states, reversed_new_states)

        # backpropagate
        for i, state, new_state in it:

            if state == new_state:
                # will break at root or the closest common state
                break

            state.num_child_walkers -= 1
            new_state.num_child_walkers += 1

            if prune and state.num_child_walkers <= 0:
                self.g.remove_node(state)
            
            # don't copy, just update reference
            self.ordered_states[i] = new_state

    @property
    def last_node(self):
        return self.ordered_states[-1]


class GameTree:
    def __init__(self, num_walkers: int, root_observation = None):
        self.num_walkers = num_walkers

        self.root = StateNode(root_observation, reward=0, num_child_walkers=self.num_walkers, terminal=False)

        self.g = nx.Graph()
        self.g.add_node(self.root)

        self.walker_paths = [Path(self.root, self.g) for _ in range(self.num_walkers)]

    def build_next_level(self, actions: Sequence, new_observations: Sequence, rewards: Sequence):
        assert len(actions) == len(new_observations) == len(rewards) == self.num_walkers

        # TODO: how can we detect duplicate observations / action transitions to save memory? (might not be super important)
        it = zip(self.walker_paths, actions, new_observations, rewards)
        for path, action, new_observation, reward in it:
            last_node = path.last_node

            # TODO: denote terminal states
            new_node = StateNode(new_observation, reward, terminal=False)
            path.add_node(new_node)

            self.g.add_edge(last_node, new_node, action=action)

    def clone(self, partners: Sequence, clone_mask: Sequence):
        for i, path in enumerate(self.walker_paths):
            if not clone_mask[i]:
                # do not clone
                continue

            target_path = self.walker_paths[partners[i]]
            path.clone_to(target_path)

    def render(self):
        colors = []
        for node in self.g.nodes:
            if node == self.root:
                colors.append("green")
            else:
                colors.append("blue")

        nx.draw(self.g, with_labels=True, node_color=colors)
        plt.show()