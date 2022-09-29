import gym
import torch

from fractal_zero.utils import get_space_distance_function


def test_dict():
    space = gym.spaces.Dict({
        "x": gym.spaces.Discrete(4),
        "y": gym.spaces.Box(low=0, high=1, shape=(2,)),
        "z": gym.spaces.Dict({
            "key": gym.spaces.Discrete(4),
        }),
    })

    space.seed(5)

    dist_func = get_space_distance_function(space)

    a0 = space.sample()
    a1 = space.sample()

    dist = dist_func(a0, a1)
    
    print(a0)
    print(a1)
    print(dist)

    assert torch.isclose(dist, torch.tensor(9.1809))