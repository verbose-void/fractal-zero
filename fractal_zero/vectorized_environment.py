from abc import ABC
from copy import deepcopy
from typing import List, Union
import gym
import ray
import torch
import numpy as np

from fractal_zero.models.joint_model import JointModel
from fractal_zero.utils import get_space_shape


def load_environment(env: Union[str, gym.Env]) -> gym.Env:
    if isinstance(env, str):
        return gym.make(env)
    return env


class VectorizedEnvironment(ABC):
    action_space: gym.Space
    observation_space: gym.Space
    n: int

    def __init__(self, env: Union[str, gym.Env], n: int):
        env = load_environment(env)
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.n = n

    def batched_action_space_sample(self):
        actions = []
        for _ in range(self.n):
            actions.append(self.action_space.sample())
        return actions

    def batch_step(self, actions):
        raise NotImplementedError

    def batch_reset(self):
        raise NotImplementedError

    def set_all_states(self, new_env: gym.Env, observation: np.ndarray):
        raise NotImplementedError

    def clone(self, partners, clone_mask):
        raise NotImplementedError


@ray.remote
class _RayWrappedEnvironment:
    def __init__(self, env: Union[str, gym.Env]):
        self._env = load_environment(env)

    def set_state(self, env: gym.Env):
        if not isinstance(env, gym.Env):
            raise ValueError(f"Expected a gym environment. Got {type(env)}.")

        self._env = deepcopy(env)

    def get_state(self) -> gym.Env:
        return self._env

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def step(self, action, *args, **kwargs):
        return self._env.step(action, *args, **kwargs)


class RayVectorizedEnvironment(VectorizedEnvironment):
    envs: List[_RayWrappedEnvironment]

    def __init__(self, env: Union[str, gym.Env], n: int):
        super().__init__(env, n)

        self.envs = [_RayWrappedEnvironment.remote(env) for _ in range(n)]

    def batch_reset(self, *args, **kwargs):
        return ray.get([env.reset.remote(*args, **kwargs) for env in self.envs])

    def batch_step(self, actions, *args, **kwargs):
        assert len(actions) == self.n

        returns = []
        for i, env in enumerate(self.envs):
            action = actions[i]
            ret = env.step.remote(action, *args, **kwargs)
            returns.append(ret)

        observations = []
        rewards = []
        dones = []
        infos = []
        for ret in ray.get(returns):
            obs, rew, done, info = ret
            observations.append(obs)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)

        # TODO: these shapes and such should be cleaner to understand and more standardized throughout the code.
        return torch.tensor(observations), torch.tensor(rewards).unsqueeze(-1), dones, infos

    def set_all_states(self, new_env: gym.Env, _):
        # NOTE: don't need to call ray.get here.
        [env.set_state.remote(new_env) for env in self.envs]

    def clone(self, partners, clone_mask):
        assert len(clone_mask) == self.n

        for i, do_clone in enumerate(clone_mask):
            if not do_clone:
                continue
            wrapped_env = self.envs[i]
            new_state = self.envs[partners[i]].get_state.remote()
            wrapped_env.set_state.remote(new_state)

class VectorizedDynamicsModelEnvironment(VectorizedEnvironment):
    def __init__(self, env: Union[str, gym.Env], n: int, joint_model: JointModel):
        super().__init__(env, n)

        if not isinstance(joint_model, JointModel):
            raise ValueError(f"Expected JointModel, got {type(joint_model)}")

        self.joint_model = joint_model

    @property
    def dynamics_model(self):
        return self.joint_model.dynamics_model

    @property
    def representation_model(self):
        return self.joint_model.representation_model

    def batch_reset(self, *args, **kwargs):
        raise NotImplementedError

    def batch_step(self, actions, *args, **kwargs):
        device = self.joint_model.device

        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=device).float().unsqueeze(-1)

        rewards = self.dynamics_model.forward(actions)
        observations = self.dynamics_model.state
        dones = torch.zeros(self.n, dtype=bool, device=device)
        infos = [{} for _ in range(self.n)]

        return observations, rewards, dones, infos

    def set_all_states(self, _, obs: np.ndarray):
        # TODO: explain, also how this interacts with FMC.

        device = self.joint_model.device

        obs = torch.tensor(obs, device=device)

        state = self.representation_model.forward(obs)

        batched_initial_state = torch.zeros(
            (self.n, *state.shape), device=device
        )
        batched_initial_state[:] = state

        self.dynamics_model.set_state(batched_initial_state)

    def clone(self, partners, clone_mask):
        state = self.dynamics_model.state
        state[clone_mask] = state[partners[clone_mask]]