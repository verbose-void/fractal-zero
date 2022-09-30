from typing import Dict
import torch
import torch.nn.functional as F

import gym
import gym.spaces as spaces


def _float_cast(vec):
    if isinstance(vec, torch.Tensor):
        return vec.float()
    return torch.tensor(vec, dtype=float).float()


def _long_cast(vec):
    if isinstance(vec, torch.Tensor):
        return vec.long()
    return torch.tensor(vec, dtype=torch.long).long()


def _cast_then_mse(y, t) -> float:
    y = _float_cast(y)
    t = _float_cast(t)
    return F.mse_loss(y, t)


def get_space_loss_function(space: gym.Space):
    # TODO: docstring

    if isinstance(space, gym.spaces.Tuple):
        raise NotImplementedError

    if isinstance(space, gym.spaces.Dict):

        # build reduce function

        funcs = {}
        for key, sub_space in space.items():
            sub_space_dist_func = get_space_loss_function(sub_space)
            funcs[key] = sub_space_dist_func

        def _composite_distance(y, t):
            # TODO: assert same space spec?

            c = 0
            total = 0
            for key, func in funcs.items():
                y_value = y[key]
                t_value = t[key]
                total += func(y_value, t_value)
                c += 1

            return total / c

        return _composite_distance

    if isinstance(space, gym.spaces.Box):
        return _cast_then_mse

    if isinstance(space, gym.spaces.Discrete):
        return _cast_then_mse

    raise NotImplementedError


class DiscreteSpaceLoss:
    def __init__(self, discrete_space: spaces.Discrete, loss_func=F.mse_loss):
        if not isinstance(discrete_space, spaces.Discrete):
            raise ValueError(f"Expected Discrete space, got {discrete_space}.")

        self.space = discrete_space
        self.loss_func = loss_func

    def _cast_x(self, x) -> torch.Tensor:
        return _float_cast(x)

    def _cast_y(self, y) -> torch.Tensor:
        if self.loss_func == F.mse_loss:
            return _float_cast(y)
        
        elif self.loss_func == F.cross_entropy:
            return _long_cast(y)

        raise NotImplementedError(f"Cast is not implemented for {self.loss_func}")

    def __call__(self, x, y):
        x = self._cast_x(x)
        y = self._cast_y(y)

        # NOTE:
        # discrete loss should be defined for the following scenarios:
        # 1. the straight up "distance" between 2 action samples
        #    for example, if a discrete space is sampled and returns `5`,
        #    and it's being compared to another discrete sample `3`,
        #    the "distance" should be `2`.
        # 2. the actual loss between an action sample and some targets?
        #    for example, when a discrete target is `2` and the input is a
        #    probability distribution `[0.2, 0.2, 0.6]`, the loss should be
        #    cross entropy where the `2` is inferred to be a class target.

        return self.loss_func(x, y)


class BoxLoss:
    def __init__(self, box_space: spaces.Box):
        if not isinstance(box_space, spaces.Box):
            raise ValueError(f"Expected Discrete space, got {box_space}.")
        self.space = box_space

    def __call__(self, x, y):
        return F.mse_loss(_float_cast(x), _float_cast(y))


class DictLoss:
    def __init__(self, dict_space: spaces.Space, loss_function_spec: Dict=None):
        if loss_function_spec is None:
            loss_function_spec = {}

        pass

    def __call__(self, y, t):
        raise NotImplementedError


class SpaceLoss:
    def __init__(self, space: gym.Space, loss_function_spec: Dict=None):
        self.space = space

        if isinstance(self.space, spaces.Box):
            self.loss_callable = BoxLoss(self.space)
        elif isinstance(self.space, spaces.Discrete):
            self.loss_callable = DiscreteSpaceLoss(self.space)
        elif isinstance(self.space, spaces.Dict):
            # TODO: include loss function spec
            raise NotImplementedError("TODO: recursively create class")
        elif isinstance(self.space, spaces.Tuple):
            raise NotImplementedError("TODO: recursively create class")
        else:
            raise NotImplementedError(f"Doesn't yet support {type(self.space)}")

        # TODO: flag to allow values outside of the ranges

    def __call__(self, x, y):
        return self.loss_callable(x, y)
