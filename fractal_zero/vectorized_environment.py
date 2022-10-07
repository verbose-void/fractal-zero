from abc import ABC
from copy import deepcopy
from typing import Callable, List, Union
import gym
import ray
import torch
import numpy as np

from fractal_zero.models.joint_model import JointModel
from fractal_zero.utils import get_space_shape


def dummy_frozen_mask(n: int):
    return torch.zeros(n).bool()


def load_environment(env: Union[str, gym.Env], copy: bool = False) -> gym.Env:
    if isinstance(env, str):
        return gym.make(env)
    if copy:
        return deepcopy(env)
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

    def batch_step(self, actions, frozen_mask=None):
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
        self.last_ret = None
        return self._env.reset(*args, **kwargs)

    def step(self, action, *args, **kwargs):
        self.last_ret = self._env.step(action, *args, **kwargs)
        return self.last_ret

    def empty_step(self):
        obs, _, done, info = self.last_ret
        return obs, 0, done, info

    def get_action_space(self):
        return self._env.action_space


class _WrappedEnvironment:
    def __init__(self, env: Union[str, gym.Env]):
        self._env = load_environment(env, copy=True)

    @property
    def action_space(self):
        return self._env.action_space

    def set_state(self, env: gym.Env):
        if not isinstance(env, gym.Env):
            raise ValueError(f"Expected a gym environment. Got {type(env)}.")

        self._env = deepcopy(env)

    def get_state(self) -> gym.Env:
        return self._env

    def reset(self, *args, **kwargs):
        self.last_ret = None
        return self._env.reset(*args, **kwargs)

    def step(self, action, *args, **kwargs):
        self.last_ret = self._env.step(action, *args, **kwargs)
        return self.last_ret

    def empty_step(self):
        obs, _, done, info = self.last_ret
        return obs, 0, done, info

    def get_action_space(self):
        return self._env.action_space


class RayVectorizedEnvironment(VectorizedEnvironment):
    envs: List[_RayWrappedEnvironment]

    def __init__(
        self, env: Union[str, gym.Env], n: int, observation_encoder: Callable = None
    ):
        super().__init__(env, n)

        # TODO: explain
        self.observation_encoder = (
            observation_encoder if observation_encoder else torch.tensor
        )

        self.envs = [_RayWrappedEnvironment.remote(env) for _ in range(n)]

    def batch_reset(self, *args, **kwargs):
        return ray.get([env.reset.remote(*args, **kwargs) for env in self.envs])

    def batch_step(self, actions, frozen_mask=None, *args, **kwargs):
        assert len(actions) == self.n

        if frozen_mask is None:
            frozen_mask = dummy_frozen_mask(self.n)

        returns = []
        for i, env in enumerate(self.envs):
            if frozen_mask[i]:
                returns.append(env.empty_step.remote())
            else:
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

        states = self.observation_encoder(observations)

        return (
            states,
            observations,
            torch.tensor(rewards, dtype=float),
            torch.tensor(dones, dtype=bool),
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


class SerialVectorizedEnvironment(VectorizedEnvironment):
    envs: List[gym.Env]

    def __init__(
        self, env: Union[str, gym.Env], n: int, observation_encoder: Callable = None
    ):
        super().__init__(env, n)

        # TODO: explain
        self.observation_encoder = (
            observation_encoder if observation_encoder else torch.tensor
        )

        self.envs = [_WrappedEnvironment(env) for _ in range(n)]

    def batch_reset(self, *args, **kwargs):
        return [env.reset(*args, **kwargs) for env in self.envs]

    def batch_step(self, actions, frozen_mask=None, *args, **kwargs):
        assert len(actions) == self.n

        if frozen_mask is None:
            frozen_mask = dummy_frozen_mask(self.n)

        observations = []
        rewards = []
        dones = []
        infos = []
        for i, env in enumerate(self.envs):
            if frozen_mask[i]:
                ret = env.empty_step()
            else:
                action = actions[i]
                ret = env.step(action, *args, **kwargs)

            obs, rew, done, info = ret
            observations.append(obs)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)

        states = self.observation_encoder(observations)

        return (
            states,
            observations,
            torch.tensor(rewards, dtype=float),
            torch.tensor(dones, dtype=bool),
            infos,
        )

    def set_all_states(self, new_env: gym.Env, obs: np.ndarray):
        self.envs = [deepcopy(new_env) for _ in range(self.n)]

    def clone(self, partners, clone_mask):
        assert len(clone_mask) == self.n

        new_envs = []
        for i, do_clone in enumerate(clone_mask):
            env = self.envs[i]
            if do_clone:
                env = deepcopy(self.envs[partners[i]])
            new_envs.append(env)
        self.envs = new_envs

    def batched_action_space_sample(self):
        actions = []
        for env in self.envs:
            action_space = env.action_space
            action_space.seed(None)  # TODO: better way to do this..
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

    def batch_step(self, actions, frozen_mask=None, *args, **kwargs):
        device = self.joint_model.device

        if frozen_mask is not None:
            raise NotImplementedError

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


# @ray.remote
# class RayEnvironmentCell(SerialVectorizedEnvironment):
#     def __init__(self, env: Union[str, gym.Env], n: int, rank: int):
#         super().__init__(env, n)
#         self.rank = rank


# class RayCellularizedEnvironment(VectorizedEnvironment):
#     cells: List[SerialVectorizedEnvironment]

#     def __init__(self, env: Union[str, gym.Env], num_environments_per_process: int, num_processes: int):
#         assert num_environments_per_process > 0
#         assert num_processes > 1

#         self.cells = []
#         for rank in range(num_processes):
#             cell = RayEnvironmentCell.remote(env, num_environments_per_process, rank)
#             self.cells.append(cell)

#         self.num_envs_per_proc = num_environments_per_process
#         self.num_processes = num_processes

#     def _gather_cellularized_returns(self, all_cell_returns):
#         flattened_returns = []
#         for actual_cell_returns in ray.get(all_cell_returns):
#             flattened_returns.extend(actual_cell_returns)
#         return flattened_returns

#     def batch_reset(self):
#         all_cell_returns = []
#         for cell in self.cells:
#             cell_returns = cell.batch_reset.remote()
#             all_cell_returns.append(cell_returns)
#         return self._gather_cellularized_returns(all_cell_returns)

#     def batched_action_space_sample(self):
#         all_cell_returns = []
#         for cell in self.cells:
#             cell_returns = cell.batched_action_space_sample.remote()
#             all_cell_returns.append(cell_returns)
#         return self._gather_cellularized_returns(all_cell_returns)

#     def _decompose_returns(self, flattened_cell_returns: List):
#         # TODO: explain
#         n_items = len(flattened_cell_returns[0])
#         items = [[] for _ in range(n_items)]

#         for flat_cell_returns in flattened_cell_returns:


#     def batch_step(self, actions: List, frozen_mask=None):
#         all_cell_returns = []
#         for rank, cell in enumerate(self.cells):
#             start = rank * self.num_envs_per_proc
#             end = start + self.num_envs_per_proc
#             cell_actions = actions[start:end]
#             cell_returns = cell.batch_step.remote(cell_actions, frozen_mask=frozen_mask)
#             all_cell_returns.append(cell_returns)
#         flattened_cell_returns = self._gather_cellularized_returns(all_cell_returns)

#         # TODO: apply observation encoder to states (?)

#         return self._decompose_returns(flattened_cell_returns)