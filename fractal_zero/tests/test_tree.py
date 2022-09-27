import numpy as np

from fractal_zero.search.tree import GameTree


def test_tree():
    n = 8
    root_observation = 0
    walker_states = np.ones(n) * root_observation

    tree = GameTree(n, root_observation=root_observation)

    # step walkers, no cloning
    actions = np.ones(n)
    walker_states += actions
    rewards = np.ones(n)

    tree.build_next_level(actions, walker_states, rewards)

    tree.render()