import gym

import torch


def get_space_shape(space):
    if isinstance(space, gym.spaces.Discrete):
        return (1,)

    if isinstance(space, gym.spaces.Box):
        return space.shape

    raise NotImplementedError(f"Type not supported: {type(space)}")


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
def calculate_virtual_rewards(exploit_vector: torch.Tensor, explore_vector: torch.Tensor, balance: float=1.0, to_cpu: bool=True):
    """Virtual rewards are a method of balancing exploration and exploitation for some purpose. The idea originates
    from FragileAI's Fractal Monte Carlo as part of the cellular automota evolution strategy, however it can be generally
    used for many purposes by replacing the exploit and explore vectors with other various metrics so this function exists
    as a generalization to be used in other places.
    """

    exploit = relativize_vector(exploit_vector) ** balance
    explore = relativize_vector(explore_vector)

    if to_cpu:
        exploit = exploit.cpu()
        explore = explore.cpu()
    
    return exploit * explore

