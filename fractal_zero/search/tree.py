

import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import List, Sequence
from uuid import UUID, uuid4


class StateNode:
    def __init__(self, observation, reward, terminal: bool = False):
        self.id = uuid4()
        self.observation = observation
        self.reward = reward
        self.terminal = terminal

    @staticmethod
    def make_root(root_observation=None, root_reward: float=0):
        return StateNode(observation=root_observation, reward=root_reward, terminal=False)

    def __str__(self) -> str:
        return f"{self.reward}"

    def __repr__(self):
        return self.__str__


class Path:
    ordered_states: List[UUID]

    def __init__(self, root: StateNode):
        self.root = root
        self.ordered_states = [root]

    def add_node(self, node: StateNode):
        self.ordered_states.append(node.id)

    def clone_to(self, other_path: "Path"):
        if self.root != other_path.root:
            raise ValueError("Cannot clone to a path unless they share the same root.")

        self.ordered_states = deepcopy(other_path.ordered_states)

    @property
    def last_node(self):
        return self.ordered_states[-1]


class GameTree:
    def __init__(self, num_walkers: int, root_observation = None):
        self.num_walkers = num_walkers

        self.root = StateNode.make_root(root_observation)
        self.walker_paths = [Path(self.root) for _ in range(self.num_walkers)]

        self.g = nx.Graph()
        self.g.add_node(self.root)

    def build_next_level(self, actions: Sequence, new_observations: Sequence, rewards: Sequence):
        assert len(actions) == len(new_observations) == len(rewards) == self.num_walkers

        # TODO
        # step 1:
        # we need to create a new layer of nodes.

        # however, we shouldn't just create a node for every walker, because some walkers may have
        # cloned to others. 
        # we need to register a new node for every *unique* state that was created before the next step.
        
        # the edge between these states should have an action.

        # TODO: how can we detect duplicate observations / action transitions to save memory?

        it = zip(self.walker_paths, actions, new_observations, rewards)
        for path, action, new_observation, reward in it:
            # TODO: terminals

            last_node = path.last_node
            node = StateNode(new_observation, reward, terminal=False)
            path.add_node(node)

            self.g.add_node(node)
            self.g.add_edge(last_node, node, action=action)
        
        self._backpropagate()

    def _backpropagate(self):
        # TODO
        # step 2:
        # when a walker moves from one state to another laterally (cloning), the update should be
        # backpropagated and pruning should occur when any node reaches a child visit count == 0.
        pass

    def render(self):
        nx.draw(self.g, with_labels=True)
        plt.show()