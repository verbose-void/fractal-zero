import gym
import gym.spaces as spaces

import torch
import torch.nn.functional as F

from fractal_zero.loss.space_loss import DiscreteSpaceLoss, SpaceLoss, get_space_loss_function


def _general_assertions(space: spaces.Space, criterion: SpaceLoss):

    # non-batched
    for _ in range(10):
        x0 = space.sample()
        x1 = space.sample()
        print(x0, x1)
        criterion(x0, x1)

    # batched
    for _ in range(10):
        x0 = [space.sample() for _ in range(10)]
        x1 = [space.sample() for _ in range(10)]
        criterion(x0, x1)

    # TODO: out of range warning?




def test_box():
    space = spaces.Box(low=0, high=2, shape=(5, 3))
    criterion = SpaceLoss(space)

    _general_assertions(space, criterion)

def test_discrete():
    space = spaces.Discrete(5)
    criterion = SpaceLoss(space)
    _general_assertions(space, criterion)

    # use cross entropy (expects logits)
    criterion = DiscreteSpaceLoss(space, loss_func=F.cross_entropy)

    # no batch
    logits = torch.tensor([0, 0.1, 0.5, 1, 1])
    target = space.sample()
    loss = criterion(logits, target)
    print("loss", loss)
    assert loss.shape == tuple()

    # batch
    blogits = torch.tensor([[0, 0.1, 0.5, 1, 1], [0, 0.1, 0.5, 1, 1], [0, 0.1, 0.5, 1, 1]])
    btarget = [space.sample() for _ in range(3)]
    bloss = criterion(blogits, btarget)
    print("bloss", bloss)
    assert bloss.shape == tuple()

    # SpaceLoss with loss func specs
    # criterion = SpaceLoss(
    #     spaces.Dict({"space0": space}), 
    #     loss_function_spec=F.cross_entropy
    # )


# def test_dict():
#     space = gym.spaces.Dict({
#         "x": gym.spaces.Discrete(4),
#         "y": gym.spaces.Box(low=0, high=1, shape=(2,)),
#         "z": gym.spaces.Dict({
#             "key": gym.spaces.Discrete(4),
#         }),
#     })

#     space.seed(5)

#     critereon = get_space_loss_function(space)

#     a0 = space.sample()
#     a1 = space.sample()

#     dist = critereon(a0, a1)
    
#     print(a0)
#     print(a1)
#     print(dist)

#     assert torch.isclose(dist, torch.tensor(9.1809))