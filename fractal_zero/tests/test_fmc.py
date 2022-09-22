import gym

from fractal_zero.config import FMCConfig
from fractal_zero.search.fmc import FMC
from fractal_zero.vectorized_environment import RayVectorizedEnvironment, VectorizedDynamicsModelEnvironment

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

    action = fmc.simulate(16)
    root_value = fmc.root_value

    assert root_value > 5

def test_cartpole_actual_environment_no_value_function():
    env = gym.make("CartPole-v0")

    vec_env = RayVectorizedEnvironment(env, n=NUM_WALKERS)
    vec_env.batch_reset()

    # no config, no model
    fmc = FMC(vec_env)

    assert fmc.config.clone_strategy != "predicted_values", "Cannot use predicted values to clone when no value function exists."

    fmc.simulate(16)
    root_value = fmc.root_value

    assert root_value > 5


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

    action = fmc.simulate(16)
    root_value = fmc.root_value

    print(action, root_value)