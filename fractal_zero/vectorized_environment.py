from abc import ABC
from typing import Union
import gym
import ray


class VectorizedEnvironment(ABC):
    action_space: gym.Space
    observation_space: gym.Space

    def batch_step(self, action):
        raise NotImplementedError

    def batch_reset(self):
        raise NotImplementedError


@ray.remote
class _RayWrappedEnvironment:
    def __init__(self, env: Union[str, gym.Env]):
        if isinstance(env, str):
            self._env = gym.make(env)
        else:
            self._env = env

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self._env.step(*args, **kwargs)



class RayVectorizedEnvironment(VectorizedEnvironment):
    def __init__(self, env: Union[str, gym.Env], n: int):
        self.envs = [_RayWrappedEnvironment.remote(env) for _ in range(n)]

    def batch_reset(self, *args, **kwargs):
        return ray.get([env.reset.remote(*args, **kwargs) for env in self.envs])

    def batch_step(self, action, *args, **kwargs):
        return ray.get([env.step.remote(action, *args, **kwargs) for env in self.envs])