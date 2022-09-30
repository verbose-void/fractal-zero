from abc import ABC
from copy import deepcopy
from typing import Callable, List, Union
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
    n: int
    action_space: gym.Space
    n: int

    def __init__(self, env: Union[str, gym.Env], n: int):
        env = load_environment(env)
        self._action_space = env.action_space
        self.n = n

    def batched_action_space_sample(self):
        actions = []
        for _ in range(self.n):
            actions.append(self._action_space.sample())
        return actions

    def __init__(self, env: Union[str, gym.Env], n: int):
        env = load_environment(env)
        self.n = n

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

    def get_action_space(self):
        return self._env.action_space

class RayVectorizedEnvironment(VectorizedEnvironment):
    envs: List[_RayWrappedEnvironment]

    def __init__(self, env: Union[str, gym.Env], n: int, observation_encoder: Callable=None):
        super().__init__(env, n)

        self.envs = [_RayWrappedEnvironment.remote(env) for _ in range(n)]
        self.observation_encoder = observation_encoder  # TODO: explain

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

        if self.observation_encoder:
            states = self.observation_encoder(observations)
        else:
            states = observations

        return (
            states,
            observations,
            torch.tensor(rewards).unsqueeze(-1).float(),
            dones,
            infos,
        )

    def set_all_states(self, new_env: gym.Env, obs: np.ndarray):
        # NOTE: don't need to call ray.get here.
        [env.set_state.remote(new_env) for env in self.envs]

    def clone(self, partners, clone_mask):
        assert len(clone_mask) == self.n

        new_envs = []

        # TODO: this kind of cloning might not be the same as vectorized cloning!!!
        for i, do_clone in enumerate(clone_mask):
            wrapped_env = self.envs[i]

            if do_clone:
                new_state = self.envs[partners[i]].get_state.remote()
                wrapped_env.set_state.remote(new_state)

            new_envs.append(wrapped_env)

        self.envs = new_envs

    def batched_action_space_sample(self):
        actions = []
        for env in self.envs:
            action_space = ray.get(env.get_action_space.remote())
            actions.append(action_space.sample())
        return actions


class VectorizedDynamicsModelEnvironment(VectorizedEnvironment):
    def __init__(self, env: Union[str, gym.Env], n: int, joint_model: JointModel):
        super().__init__(env, n)

        self._env = env

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
        obs = self._env.reset(*args, **kwargs)
        self.set_all_states(self._env, obs)
        return obs

    def batch_step(self, actions, *args, **kwargs):
        device = self.joint_model.device

        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=device).float().unsqueeze(-1)

        rewards = self.dynamics_model.forward(actions)
        observations = self.dynamics_model.state
        dones = torch.zeros(self.n, dtype=bool, device=device)
        infos = [{} for _ in range(self.n)]

        return observations, rewards, dones, infos

    def set_all_states(self, new_env: gym.Env, obs: np.ndarray):
        # TODO: explain, also how this interacts with FMC.

        device = self.joint_model.device

        obs = torch.tensor(obs, device=device)

        state = self.representation_model.forward(obs)

        batched_initial_state = torch.zeros((self.n, *state.shape), device=device)
        batched_initial_state[:] = state

        self.dynamics_model.set_state(batched_initial_state)

    def clone(self, partners, clone_mask):
        state = self.dynamics_model.state
        state[clone_mask] = state[partners[clone_mask]]