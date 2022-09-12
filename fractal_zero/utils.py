import gym

import numpy as np
import torch


def get_space_shape(space):
    if isinstance(space, gym.spaces.Discrete):
        return (1,)

    if isinstance(space, gym.spaces.Box):
        return space.shape

    raise NotImplementedError(f"Type not supported: {type(space)}")


@torch.no_grad()
def mean_min_max_dict(name: str, arr) -> dict:
    if isinstance(arr, list):
        arr = torch.tensor(arr, dtype=float)

    return {
        f"{name}/mean": arr.mean(),
        f"{name}/min": arr.min(),
        f"{name}/max": arr.max(),
    }


@torch.no_grad()
def relativize_vector(vector: torch.Tensor):
    # TODO: docstring

    std = vector.std()
    if std == 0:
        return torch.ones(len(vector))
    standard = (vector - vector.mean()) / std
    standard[standard > 0] = torch.log(1 + standard[standard > 0]) + 1
    standard[standard <= 0] = torch.exp(standard[standard <= 0])
    return standard


@torch.no_grad()
def determine_partners(n: int):
    # TODO: docstring

    choices = np.random.choice(np.arange(n), size=n)
    return torch.tensor(choices, dtype=int)


@torch.no_grad()
def calculate_distances(vector: torch.Tensor, partners: torch.Tensor, dim: int=1):
    assert len(vector.shape) == 2
    assert len(partners.shape) == 1
    return torch.linalg.norm(
        vector - vector[partners], dim=dim
    )


@torch.no_grad()
def calculate_virtual_rewards(exploit_vector: torch.Tensor, explore_vector: torch.Tensor, balance: float=1.0, to_cpu: bool=True, softmax: bool=False, dim: int=None):
    """Virtual rewards are a method of balancing exploration and exploitation for some purpose. The idea originates
    from FragileAI's Fractal Monte Carlo as part of the cellular automota evolution strategy, however it can be generally
    used for many purposes by replacing the exploit and explore vectors with other various metrics so this function exists
    as a generalization to be used in other places.
    """

    if len(exploit_vector.shape) != len(explore_vector.shape):
        raise ValueError(f"Shapes for exploit ({exploit_vector.shape}) and explore ({explore_vector.shape}) must match. ")

    exploit = relativize_vector(exploit_vector) ** balance
    explore = relativize_vector(explore_vector)

    if to_cpu:
        exploit = exploit.cpu()
        explore = explore.cpu()
    
    virtual_rewards = exploit * explore

    if softmax:
        torch.nn.functional.softmax(virtual_rewards, dim=dim)

    return virtual_rewards

