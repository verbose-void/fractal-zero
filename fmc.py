
import torch


def get_random_actions(dynamics_model, n: int):
    actions = []
    for _ in range(n):
        action = dynamics_model.action_space.sample()
        actions.append(action)
    return torch.tensor(actions).unsqueeze(-1)


def lookahead(initial_state, dynamics_model, k: int, num_walkers: int = 4):
    action_history = []
    reward_history = []

    state = torch.zeros((num_walkers, *initial_state.shape))
    state[:] = initial_state

    dynamics_model.set_state(state)

    for _ in range(k):
        actions = get_random_actions(dynamics_model, num_walkers)
        print(state.shape, actions.shape)

        rewards = dynamics_model.forward(actions)

        action_history.append(actions)
        reward_history.append(rewards)

        print(rewards.shape)

    # TODO: select the best action
    return action_history[0][0, 0].numpy()