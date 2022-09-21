import gym

from fractal_zero.config import FractalZeroConfig
from fractal_zero.search.fmc import FMC
from fractal_zero.vectorized_environment import RayVectorizedEnvironment

from fractal_zero.tests.test_vectorized_environment import build_test_joint_model


def test_cartpole():
    n = 8
    alphazero_style = True

    env = gym.make("CartPole-v0")
    joint_model = build_test_joint_model(env, embedding_size=4)

    # TODO: clean API, FMC should *not* require this config to be general.
    config = FractalZeroConfig(
        env,
        joint_model,
        search_using_actual_environment=alphazero_style,
        max_replay_buffer_size=64,
        num_games=1_024,
        max_game_steps=200,
        max_batch_size=16,
        unroll_steps=8,
        learning_rate=0.003,
        optimizer="SGD",
        weight_decay=1e-4,
        momentum=0.9,  # only if optimizer is SGD
        num_walkers=n,
        balance=1.0,
        lookahead_steps=8,
        evaluation_lookahead_steps=8,
        # wandb_config={"project": "fractal_zero_cartpole"},
    )

    vec_env = RayVectorizedEnvironment(env, n=n)
    vec_env.batch_reset()

    fmc = FMC(vec_env, config, verbose=False)

    action = fmc.simulate(16)
    root_value = fmc.root_value

    assert root_value > 5