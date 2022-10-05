import gym
import numpy as np
import networkx as nx

# from fractal_zero.search.fmc import FMC
from fractal_zero.search.fmc_new import FMC
from fractal_zero.vectorized_environment import (
    RayVectorizedEnvironment,
    SerialVectorizedEnvironment,
    VectorizedDynamicsModelEnvironment,
)

import pytest


with_vec_envs = pytest.mark.parametrize(
    "vec_env_class", [SerialVectorizedEnvironment, RayVectorizedEnvironment]
)


def _assert_mean_total_rewards(fmc, steps, expected_mean_reward, use_tqdm=False, trials=8):
    total_rewards = []

    for _ in range(trials):
        fmc.reset()

        # 200 is the max reward accumulate-able in cartpole.
        fmc.simulate(steps, use_tqdm=use_tqdm)

        total_reward = fmc.tree.best_path.total_reward
        total_rewards.append(total_reward)
    assert np.mean(total_rewards) > expected_mean_reward


@with_vec_envs
def test_cartpole_actual_environment(vec_env_class):
    env = gym.make("CartPole-v0")

    n = 16
    vec_env = vec_env_class(env, n=n)
    fmc = FMC(vec_env)
    _assert_mean_total_rewards(fmc, 64, 50)


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
    n = 16
    vec_env = SerialVectorizedEnvironment("CartPole-v0", n=n)
    fmc = FMC(vec_env, balance=1)
    _assert_mean_total_rewards(fmc, 400, 175, use_tqdm=True)
