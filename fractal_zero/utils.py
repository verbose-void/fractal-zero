import gym

import torch
import torch.nn.functional as F


def kl_divergence_of_model_paramters(p0, p1):
    total = 0
    c = 0
    for param0, param1 in zip(p0, p1):
        total += F.kl_div(param0, param1)
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


def _cast_then_mse(y, t) -> float:
    return F.mse_loss(y.float(), t.float())


def get_space_distance_function(space: gym.Space):
    # TODO: docstring

    if isinstance(space, gym.spaces.Tuple):
        raise NotImplementedError

    if isinstance(space, gym.spaces.Dict):
        raise NotImplementedError

    if isinstance(space, gym.spaces.Box):
        return F.mse_loss

    if isinstance(space, gym.spaces.Discrete):
        return _cast_then_mse

    raise NotImplementedError
