from fractal_zero.models.dynamics import FullyConnectedDynamicsModel
from fractal_zero.models.joint_model import JointModel
from fractal_zero.models.prediction import FullyConnectedPredictionModel
from fractal_zero.models.representation import FullyConnectedRepresentationModel
from fractal_zero.vectorized_environment import (
    RayVectorizedEnvironment,
    VectorizedDynamicsModelEnvironment,
    VectorizedEnvironment,
    load_environment,
)

import gym


def build_test_joint_model(env, embedding_size: int = 8) -> JointModel:
    env = load_environment(env)

    rep = FullyConnectedRepresentationModel(env, embedding_size)
    dyn = FullyConnectedDynamicsModel(env, embedding_size, 1)
    pred = FullyConnectedPredictionModel(env, embedding_size)

    return JointModel(rep, dyn, pred)


def _test_step(vec_env: VectorizedEnvironment):
    actions = vec_env.batched_action_space_sample()
    obss, rews, dones, infos = vec_env.batch_step(actions)

    assert len(obss) == vec_env.n
    assert len(obss) == len(rews) == len(dones) == len(infos)


def test_vectorized_cartpole_ray():
    n = 2
    env_id = "CartPole-v0"

    vec_env = RayVectorizedEnvironment(env_id, n=n)

    initial_observations = vec_env.batch_reset()
    assert len(initial_observations) == n

    _test_step(vec_env)

    # test setting states
    new_env = gym.make(env_id)
    new_env.reset()
    for _ in range(5):
        obs, rew, done, info = new_env.step(new_env.action_space.sample())
    vec_env.set_all_states(new_env, obs)


def test_vectorized_cartpole_dynamics_model():
    n = 2

    env = load_environment("CartPole-v0")

    joint_model = build_test_joint_model(env)
    vec_env = VectorizedDynamicsModelEnvironment(env, n=n, joint_model=joint_model)

    obs = env.reset()
    vec_env.set_all_states(env, obs)

    _test_step(vec_env)
