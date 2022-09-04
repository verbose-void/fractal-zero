import gym

from latent_env import LatentEnv



def lookahead(latent_env: LatentEnv, n: int):
    actions = []
    for _ in range(n):
        action = latent_env.action_space.sample()
        latent_env.step(action)

        actions.append(action)
    return actions[0]
    

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env.reset()

    latent_env = LatentEnv(observation_space=env.observation_space, action_space=env.action_space)


    lookahead_steps = 10
    steps = 10
    for _ in range(steps):
        action = lookahead(latent_env, lookahead_steps)
        obs, reward, done, info = env.step(action)
        print("reward", reward)
        if done:
            print('done')
            break
