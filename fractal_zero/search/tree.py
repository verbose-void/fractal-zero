

from copy import deepcopy
from typing import List
from uuid import UUID, uuid4


class StateNode:
    def __init__(self, observation, reward, terminal: bool = False):
        self.id = uuid4()
        self.observation = observation
        self.reward = reward
        self.terminal = terminal

    @staticmethod
    def make_root():
        return StateNode(observation=None, reward=0)


class Path:
    ordered_states: List[UUID]

    def __init__(self, root_id: UUID):
        self.root_id = root_id
        self.ordered_states = [root_id]

    def clone_to(self, other_path: "Path"):
        if self.root_id != other_path.root_id:
            raise ValueError("Cannot clone to a path unless they share the same root.")

        self.ordered_states = deepcopy(other_path.ordered_states)


class GameTree:
    def __init__(self, num_walkers: int):
        self.num_walkers = num_walkers

        self.root = StateNode.make_root()
        self.walker_paths = [Path(self.root.id) for _ in range(self.num_walkers)]

    def build_next_level(self):
        # TODO
        # step 1:
        # we need to create a new layer of nodes.

        # however, we shouldn't just create a node for every walker, because some walkers may have
        # cloned to others. 
        # we need to register a new node for every *unique* state that was created before the next step.
        
        # the edge between these states should have an action.
        pass

    def backpropagate(self):
        # TODO
        # step 2:
        # when a walker moves from one state to another laterally (cloning), the update should be
        # backpropagated and pruning should occur when any node reaches a child visit count == 0.
        pass