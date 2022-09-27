import numpy as np

from fractal_zero.search.tree import GameTree


def test_tree_no_clone():
    n = 8
    root_observation = 0
    walker_states = np.ones(n) * root_observation

    def _step(walker_states):
        actions = np.ones(n)
        walker_states += actions
        rewards = np.ones(n)
        return actions, rewards

    tree = GameTree(n, root_observation=root_observation)
    for _ in range(4):
        actions, rewards = _step(walker_states)
        tree.build_next_level(actions, walker_states, rewards)
        tree.render()

    # TODO: test walker counts