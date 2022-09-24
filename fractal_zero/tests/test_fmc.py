import gym
import numpy as np
from tqdm import tqdm
import torch

from fractal_zero.config import FMCConfig
from fractal_zero.search.fmc import FMC
from fractal_zero.vectorized_environment import (
    RayVectorizedEnvironment,
    VectorizedDynamicsModelEnvironment,
)

from fractal_zero.tests.test_vectorized_environment import build_test_joint_model


NUM_WALKERS = 4


def test_cartpole_actual_environment():
    alphazero_style = True

    env = gym.make("CartPole-v0")
    model = build_test_joint_model(env, embedding_size=4).prediction_model

    config = FMCConfig(
        num_walkers=NUM_WALKERS,
        search_using_actual_environment=alphazero_style,
    )

    vec_env = RayVectorizedEnvironment(env, n=NUM_WALKERS)
    vec_env.batch_reset()

    fmc = FMC(vec_env, model, config, verbose=False)

    fmc.simulate(16)
    assert fmc.root_value > 5


def test_cartpole_actual_environment_no_value_function():
    env = gym.make("CartPole-v0")

    vec_env = RayVectorizedEnvironment(env, n=NUM_WALKERS)
    vec_env.batch_reset()

    # no config, no model
    fmc = FMC(vec_env)

    assert (
        fmc.config.clone_strategy != "predicted_values"
    ), "Cannot use predicted values to clone when no value function exists."

    fmc.simulate(16)

    assert fmc.root_value > 5


def test_cartpole_dynamics_function():
    alphazero_style = False

    env = gym.make("CartPole-v0")
    joint_model = build_test_joint_model(env, embedding_size=4)

    config = FMCConfig(
        num_walkers=NUM_WALKERS,
        search_using_actual_environment=alphazero_style,
    )

    vec_env = VectorizedDynamicsModelEnvironment(env, NUM_WALKERS, joint_model)
    vec_env.batch_reset()

    fmc = FMC(vec_env, prediction_model=joint_model.prediction_model, config=config)
    fmc.simulate(16)


def test_cartpole_exact_reward_and_values():
    env = gym.make("CartPole-v0")
    vec_env = RayVectorizedEnvironment(env, n=NUM_WALKERS)
    vec_env.batch_reset()

    config = FMCConfig(gamma=1, num_walkers=NUM_WALKERS, clone_strategy="cumulative_reward")
    fmc = FMC(vec_env, config=config)

    # no walkers will die from this, and the cumulative rewards/value estimation should be exact.
    fmc.simulate(8, use_tqdm=True)
    cumulative_rewards = fmc.reward_buffer.sum(dim=1)
    assert cumulative_rewards.shape == (NUM_WALKERS, 1)
    assert fmc.reward_buffer.sum() == NUM_WALKERS * 8
    np.testing.assert_almost_equal(fmc.root_value, 8)


def test_cartpole_consistently_high_reward():
    n = 64

    env = gym.make("CartPole-v0")
    vec_env = RayVectorizedEnvironment(env, n=n)

    config = FMCConfig(gamma=1, num_walkers=n, clone_strategy="cumulative_reward")
    fmc = FMC(vec_env, config=config)

    max_rewards = []
    root_values = []

    num_trials = 8
    for _ in range(num_trials):
        vec_env.batch_reset()
        fmc.reset()

        # 200 is the max reward accumulate-able in cartpole.
        fmc.simulate(200, use_tqdm=True)

        max_reward = fmc.reward_buffer.sum(dim=1).max()
        max_rewards.append(max_reward)
        root_values.append(fmc.root_value)

    # somewhat lenient constraints
    # NOTE: random max rewards average is usually <<50.
    print(max_rewards)
    print(root_values)
    assert np.mean(max_rewards) > 120
    assert np.mean(root_values) > 100


def test_cartpole_with_value_function_temporal_difference_learning():
    n = 64

    env = gym.make("CartPole-v0")
    vec_env = RayVectorizedEnvironment(env, n=n)

    # in order to train the value network, the cloning exploitation will be based on the cumulative reward (aka advantage)
    # instead of value estimates. this is just to guarentee good samples for the value function to be determined for simplicity
    # and consistency of test
    config = FMCConfig(gamma=0.99, num_walkers=n, clone_strategy="cumulative_reward")
    fmc = FMC(vec_env, config=config)

    # TODO: TD train value function, then evaluate how FMC performs.
    
    num_games = 2
    max_steps = 10
    lookahead_steps = 2

    observations = []
    value_targets = []

    # NOTE: because we are using the cumulative reward clone exploitation strategy, training the value network 
    # while generating the data will have no impact on performance.
    for _ in tqdm(range(num_games), desc="Playing games and generating data"):
        obs = env.reset()

        for _ in tqdm(range(max_steps), desc="Playing game"):
            fmc.reset()
            vec_env.set_all_states(env, obs)
            fmc_selected_action = fmc.simulate(lookahead_steps)

            observations.append(torch.tensor([obs]))
            value_targets.append(fmc.root_value)

            obs, reward, done, info = env.step(fmc_selected_action)

            if done:
                break


    # NOTE: the value estimates after training will be relative to the number of lookahead steps
    # that was used during the generation of the data.
    value_targets = torch.tensor(value_targets) / lookahead_steps  # NOTE: normalized
    observations = torch.cat(observations)

    # NOTE: assuming the value can be derived directly from a observation -> value mapping (no
    # context of surrounding observations)

    # epochs = 2
    # for _ in tqdm(range(epochs), desc="TD learning value function"):
    #     pass