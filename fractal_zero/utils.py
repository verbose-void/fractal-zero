import gym

import torch
import torch.nn.functional as F


def parameters_norm(parameters):
    c = 0
    total = 0
    for param in parameters:
        total += torch.linalg.norm(param)
        c += 1
    return total / c


def dist_of_model_paramters(p0, p1):
    total = 0
    c = 0
    for param0, param1 in zip(p0, p1):
        p0 = param0.data.flatten()
        p1 = param1.data.flatten()
        total += torch.linalg.norm(p0 - p1)  # euclidean distance
        c += 1
    return total / c


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
