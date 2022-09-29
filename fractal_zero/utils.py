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


def _float_cast(vec):
    if isinstance(vec, torch.Tensor):
        return vec.float()
    return torch.tensor(vec, dtype=float).float()


def _cast_then_mse(y, t) -> float:
    y = _float_cast(y)
    t = _float_cast(t)
    return F.mse_loss(y, t)


def get_space_distance_function(space: gym.Space):
    # TODO: docstring

    if isinstance(space, gym.spaces.Tuple):
        raise NotImplementedError

    if isinstance(space, gym.spaces.Dict):

        # build reduce function

        funcs = {}
        for key, sub_space in space.items():
            sub_space_dist_func = get_space_distance_function(sub_space)
            funcs[key] = sub_space_dist_func

        def _composite_distance(y, t):
            # TODO: assert same space spec?

            total = 0
            for key, func in funcs.items():
                y_value = y[key]
                t_value = t[key]
                total += func(y_value, t_value)

            return total  # TODO: normalize?

        return _composite_distance

    if isinstance(space, gym.spaces.Box):
        return _cast_then_mse

    if isinstance(space, gym.spaces.Discrete):
        return _cast_then_mse

    raise NotImplementedError
