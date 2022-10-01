import gym
import numpy as np
import networkx as nx

from fractal_zero.config import FMCConfig
from fractal_zero.search.fmc import FMC
from fractal_zero.vectorized_environment import (
    RayVectorizedEnvironment,
    SerialVectorizedEnvironment,
    VectorizedDynamicsModelEnvironment,
)

from fractal_zero.tests.test_vectorized_environment import build_test_joint_model
import pytest


NUM_WALKERS = 16


with_vec_envs = pytest.mark.parametrize(
    "vec_env_class", [SerialVectorizedEnvironment, RayVectorizedEnvironment]
)


@with_vec_envs
def test_cartpole_actual_environment(vec_env_class):
    env = gym.make("CartPole-v0")

    config = FMCConfig(
        num_walkers=NUM_WALKERS,
    )

    vec_env = vec_env_class(env, n=NUM_WALKERS)

    trials = 10
    for _ in range(trials):
        vec_env.batch_reset()
        fmc = FMC(vec_env, config=config, verbose=False)
        fmc.simulate(16)
        assert nx.is_tree(fmc.tree.g)
        assert fmc.tree.best_path.total_reward >= 16


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

    env = gym.make("CartPole-v0")
    vec_env = SerialVectorizedEnvironment(env, n=n)

    config = FMCConfig(num_walkers=n, clone_strategy="cumulative_reward")
    fmc = FMC(vec_env, config=config)

    total_rewards = []

    num_trials = 8
    for _ in range(num_trials):
        vec_env.batch_reset()
        fmc.reset()

        # 200 is the max reward accumulate-able in cartpole.
        fmc.simulate(200, use_tqdm=True)

        total_reward = fmc.tree.best_path.total_reward
        total_rewards.append(total_reward)

    # somewhat lenient constraints
    # NOTE: random max rewards average is usually <<50.
    assert np.mean(total_rewards) > 120
