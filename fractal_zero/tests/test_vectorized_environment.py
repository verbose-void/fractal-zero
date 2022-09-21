


from fractal_zero.vectorized_environment import RayVectorizedEnvironment


def test_vectorized_cartpole_ray():
    n = 2

    vec_env = RayVectorizedEnvironment("CartPole-v0", n=n)
    
    initial_observations = vec_env.batch_reset()
    assert len(initial_observations) == n

    actions = vec_env.batched_action_space_sample()
    print(actions)
    obss, rews, dones, infos = vec_env.batch_step(actions)

    assert len(obss) == n
    assert len(obss) == len(rews) == len(dones) == len(infos)