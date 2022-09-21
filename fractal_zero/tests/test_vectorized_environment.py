


from fractal_zero.models.dynamics import FullyConnectedDynamicsModel
from fractal_zero.models.joint_model import JointModel
from fractal_zero.models.prediction import FullyConnectedPredictionModel
from fractal_zero.models.representation import FullyConnectedRepresentationModel
from fractal_zero.vectorized_environment import RayVectorizedEnvironment, VectorizedDynamicsModelEnvironment, load_environment


def _build_test_joint_model(env) -> JointModel:
    env = load_environment(env)

    rep = FullyConnectedRepresentationModel(env, 8)
    dyn = FullyConnectedDynamicsModel(env, 8, 1)
    pred = FullyConnectedPredictionModel(env, 8)

    return JointModel(rep, dyn, pred)


def test_vectorized_cartpole_ray():
    n = 2

    vec_env = RayVectorizedEnvironment("CartPole-v0", n=n)
    
    initial_observations = vec_env.batch_reset()
    assert len(initial_observations) == n

    actions = vec_env.batched_action_space_sample()
    obss, rews, dones, infos = vec_env.batch_step(actions)

    assert len(obss) == n
    assert len(obss) == len(rews) == len(dones) == len(infos)


def test_vectorized_cartpole_dynamics_model():
    n = 2

    joint_model = _build_test_joint_model("CartPole-v0")
    vec_env = VectorizedDynamicsModelEnvironment("CartPole-v0", n=n, joint_model=joint_model)

    print(vec_env)

