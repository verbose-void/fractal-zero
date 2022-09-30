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
    def __init__(self, discrete_space: spaces.Discrete, loss_func=None):
        self.loss_func = loss_func if loss_func else F.mse_loss

        if not isinstance(discrete_space, spaces.Discrete):
            raise ValueError(f"Expected Discrete space, got {discrete_space}.")

        self.space = discrete_space

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


class BoxSpaceLoss:
    def __init__(self, box_space: spaces.Box, loss_func=None):
        self.loss_func = loss_func if loss_func else F.mse_loss
        if not isinstance(box_space, spaces.Box):
            raise ValueError(f"Expected Discrete space, got {box_space}.")
        self.space = box_space

    def __call__(self, x, y):
        return self.loss_func(_float_cast(x), _float_cast(y))


LOSS_CLASSES = {
    spaces.Discrete: DiscreteSpaceLoss,
    spaces.Box: BoxSpaceLoss,
}


class DictSpaceLoss:
    def __init__(self, dict_space: spaces.Space, loss_spec: Dict=None):
        self.space = dict_space
        self.loss_spec = loss_spec if loss_spec else {}
        self._build_funcs()

    def _build_funcs(self):
        self.funcs = {}
        for key, subspace in self.space.items():
            subspace_loss_func = self.loss_spec.get(key, None)

            if isinstance(subspace, spaces.Dict):
                subspace_loss_class = DictSpaceLoss
                subspace_kwargs = {"loss_spec": subspace_loss_func}
            elif type(subspace) not in LOSS_CLASSES:
                raise ValueError(f"Subspace \"{subspace}\" is not supported.")
            else:
                subspace_loss_class = LOSS_CLASSES[type(subspace)]
                subspace_kwargs = {"loss_func": subspace_loss_func}
                
            subspace_criterion = subspace_loss_class(subspace, **subspace_kwargs)
            self.funcs[key] = subspace_criterion

    def __call__(self, y, t):
        # TODO: support batch and reduce?

        loss = 0
        for key, func in self.funcs.items():
            loss += func(y[key], t[key])
        return loss

class SpaceLoss:
    def __init__(self, space: gym.Space, loss_function_spec: Dict=None):
        self.space = space

        if isinstance(self.space, spaces.Box):
            self.loss_callable = BoxSpaceLoss(self.space)
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
