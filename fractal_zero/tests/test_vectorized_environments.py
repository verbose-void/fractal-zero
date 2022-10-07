

from fractal_zero.vectorized_environment import RayCellularizedEnvironment


def test_cellularized_environments():
    n_env_per_proc = 4
    n_proc = 2
    n = n_env_per_proc * n_proc

    cell_env = RayCellularizedEnvironment(
        "CartPole-v0", 
        num_environments_per_process=n_env_per_proc, 
        num_processes=n_proc,
    )
    obss = cell_env.batch_reset()
    assert len(obss) == n

    for _ in range(4):
        actions = cell_env.batched_action_space_sample()
        states, obss, rewards, dones, infos = cell_env.batch_step(actions)
        assert len(states) == len(obss) == len(rewards) == len(dones) == len(infos) == n