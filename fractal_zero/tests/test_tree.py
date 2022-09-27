import numpy as np

from fractal_zero.search.tree import GameTree


def test_tree():
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
        # tree.render()
        assert tree.root.num_child_walkers == n

    # all partners with 0th walker
    partners = np.zeros(n, dtype=int)  

    # only 1 walker clones to the 0th walker
    clone_mask = np.zeros(n, dtype=bool)
    clone_mask[1] = 1

    tree.clone(partners, clone_mask)
    # tree.render()
    assert tree.g.degree(tree.root) == n - 1  # 1 walker was fully cloned away and pruned.
    assert tree.root.num_child_walkers == n
