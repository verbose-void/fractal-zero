import gym
import numpy as np
from tqdm import tqdm

# from fractal_zero.search.fmc import FMC
from fractal_zero.search.fmc import FMC
from fractal_zero.vectorized_environment import (
    RayVectorizedEnvironment,
    SerialVectorizedEnvironment,
    VectorizedDynamicsModelEnvironment,
)

import pytest


with_vec_envs = pytest.mark.parametrize(
    "vec_env_class", [SerialVectorizedEnvironment, RayVectorizedEnvironment]
)

def _check_last_actions(fmc: FMC):
    last_actions = fmc.tree.last_actions
    print("tree", last_actions)
    print("fmc", fmc.actions)
    # print("clone mask", fmc.clone_mask)
    for last_action, expected_action in zip(last_actions, fmc.actions):
        assert last_action == expected_action

def _assert_tree_equivalence(fmc: FMC):
    _check_last_actions(fmc)

    # check rewards are properly matching scores
    total_rewards = np.array([p.total_reward for p in fmc.tree.walker_paths])
    expected_total_rewards = fmc.scores.numpy()
    np.testing.assert_allclose(total_rewards, expected_total_rewards)

    # TODO: make sure the tree's best path gets the best walker's same score.
    best_walker_index = fmc.scores.argmax()
    score = fmc.scores[best_walker_index]

    expected_best_walker_path = fmc.tree.walker_paths[best_walker_index]
    assert np.isclose(score, expected_best_walker_path.total_reward)
    assert expected_best_walker_path == fmc.tree.best_path


def _assert_mean_total_rewards(
    fmc: FMC, steps, expected_mean_reward, use_tqdm=False, trials=8
):
    total_rewards = []

    for _ in range(trials):
        fmc.reset()

        # 1 step at a time
        for _ in tqdm(range(steps), disable=not use_tqdm):
            fmc.simulate(1)
            _assert_tree_equivalence(fmc)
            # cartpole has a max reward of 200.
            assert (fmc.scores <= 200).all()

            if fmc.did_early_exit:
                break

        total_reward = fmc.tree.best_path.total_reward
        total_rewards.append(total_reward)

    assert np.mean(total_rewards) > expected_mean_reward


@with_vec_envs
def test_cartpole_actual_environment(vec_env_class):
    env = gym.make("CartPole-v0")

    n = 4
    vec_env = vec_env_class(env, n=n)
    fmc = FMC(vec_env)
    _assert_mean_total_rewards(fmc, 64, 50, use_tqdm=True)


# def test_cartpole_dynamics_function():
#     alphazero_style = False

#     env = gym.make("CartPole-v0")
#     joint_model = build_test_joint_model(env, embedding_size=4)

#     config = FMCConfig(
#         num_walkers=NUM_WALKERS,
#         search_using_actual_environment=alphazero_style,
#     )

#     vec_env = VectorizedDynamicsModelEnvironment(env, NUM_WALKERS, joint_model)
#     vec_env.batch_reset()

#     fmc = FMC(vec_env, config=config)
#     fmc.simulate(16)


def test_cartpole_consistently_high_reward():
    n = 64
    vec_env = SerialVectorizedEnvironment("CartPole-v0", n=n)
    fmc = FMC(vec_env, balance=1)

    # 200 is the max reward accumulate-able in cartpole.
    _assert_mean_total_rewards(fmc, 400, 140, use_tqdm=True)
