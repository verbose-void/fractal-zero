import numpy as np
import torch

from fractal_zero.config import FractalZeroConfig
from fractal_zero.utils import calculate_distances, calculate_virtual_rewards, determine_partners


class GameHistory:
    def __init__(self):
        # first "frame" is always empty (with the actual initial observation)
        self.actions = []
        self.observations = []
        self.environment_reward_signals = []
        self.values = []

        # TODO: finalize game histories into tensors after all frames have been added?

    @property
    def observation_shape(self):
        return self.observations[0].shape

    @property
    def action_shape(self):
        return tuple()  # TODO!

    def append(self, action, observation, environment_reward_signal, value):
        self.actions.append(action)
        self.observations.append(observation)
        self.environment_reward_signals.append(environment_reward_signal)
        self.values.append(value)

    def __getitem__(self, index):
        return (
            self.observations[index],
            self.actions[index],
            self.environment_reward_signals[index],
            self.values[index],
        )

    def __len__(self):
        if (
            len(self.observations)
            == len(self.actions)
            == len(self.environment_reward_signals)
        ):
            return len(self.observations)
        raise ValueError(str(self))

    def __str__(self):
        return f"GameHistory(frames={len(self)})"


class ReplayBuffer:
    def __init__(self, config: FractalZeroConfig):
        # TODO: prioritized experience replay (PER) https://arxiv.org/abs/1511.05952

        self.config = config

        self.game_histories = []

        # TODO: move these calculations somewhere else
        self.exploit = None
        self.explore = None
        self.virtual_rewards = None

    def _maybe_pop_one(self, replacement_game_history: GameHistory):
        # TODO: docstring

        if len(self) < self.config.max_replay_buffer_size:
            return

        strat = self.config.replay_buffer_pop_strategy

        if strat == "oldest":
            i = 0
        elif strat == "random":
            i = np.random.randint(0, len(self))
        elif strat == "balanced":
            obs = replacement_game_history[-1][0]

            self.exploit = -torch.tensor(self.get_episode_lengths(), dtype=float)
            self.explore = -self._get_episode_distances(obs)
            self.virtual_rewards = calculate_virtual_rewards(self.exploit, self.explore, softmax=True)
            i = np.random.choice(range(len(self)), p=self.virtual_rewards)
        else:
            raise NotImplementedError(
                f'Replay buffer pop strategy "{strat}" is not supported.'
            )

        self.game_histories.pop(i)

    def append(self, game_history: GameHistory):
        """Add a trajectory/episode to the replay buffer. If the buffer is full, a trajectory will be popped according
        to the pop strategy specified in the config.
        """

        self._maybe_pop_one(game_history)
        self.game_histories.append(game_history)

        if len(self) > self.config.max_replay_buffer_size:
            raise ValueError

    def sample_game(self) -> GameHistory:
        game_index = np.random.randint(0, len(self.game_histories))
        return self.game_histories[game_index]

    def sample_game_clip(
        self, clip_length: int, pad_to_num_frames: bool = True
    ) -> tuple:
        assert clip_length > 0

        game = self.sample_game()

        # minimizing padding means the start frame chosen will result in the least amount of padded frames.
        if self.config.minimize_batch_padding:
            if len(game) <= clip_length:
                start_frame = 0
            else:
                start_frame = np.random.randint(0, len(game) - clip_length)
        else:
            start_frame = np.random.randint(0, len(game))

        end_frame = start_frame + clip_length

        actual_frames = game[start_frame:end_frame]

        actual_num_frames = len(actual_frames[0])
        num_frames = clip_length if pad_to_num_frames else actual_num_frames

        observations = np.zeros((num_frames, *game.observation_shape), dtype=float)
        actions = np.zeros(
            (
                num_frames,
                *game.action_shape,
            ),
            dtype=float,
        )
        rewards = np.zeros((num_frames,), dtype=float)
        values = np.zeros((num_frames,), dtype=float)

        observations[:actual_num_frames] = actual_frames[0]
        actions[:actual_num_frames] = actual_frames[1]
        rewards[:actual_num_frames] = actual_frames[2]
        values[:actual_num_frames] = actual_frames[3]

        num_empty_frames = clip_length - actual_num_frames

        return observations, actions, rewards, values, num_empty_frames

    def get_episode_lengths(self):
        return [len(history) for history in self.game_histories]

    def _get_episode_distances(self, ref_obs: np.ndarray = None):
        # TODO: docstring, this is a dense function.

        n = len(self) if ref_obs is None else len(self) + 1
        last_observations = np.zeros((n, *self.config.observation_shape))
        
        for i in range(len(self)):
            last_observations[i] = self[i][-1][0]

        if ref_obs is not None:
            last_observations[-1] = ref_obs
            partners = -torch.ones(n).long()
        else:
            partners = determine_partners(len(self))

        dists = calculate_distances(torch.tensor(last_observations, dtype=float), partners)
        return dists if ref_obs is None else dists[:-1]

    def __getitem__(self, index):
        return self.game_histories[index]

    def __len__(self):
        return len(self.game_histories)
