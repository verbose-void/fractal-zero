from copy import deepcopy
from typing import Callable, List
import torch

from tqdm import tqdm
from fractal_zero.search.tree import GameTree

from fractal_zero.vectorized_environment import VectorizedEnvironment


def _l2_distance(vec0, vec1):
    assert len(vec0.shape) <= 2, "Not defined for dimensionality > 2."
    return torch.norm(vec0 - vec1, dim=-1)


def _relativize_vector(vector: torch.Tensor):
    std = vector.std()
    if std == 0:
        return torch.ones(len(vector))
    standard = (vector - vector.mean()) / std
    standard[standard > 0] = torch.log(1 + standard[standard > 0]) + 1
    standard[standard <= 0] = torch.exp(standard[standard <= 0])
    return standard


_ATTRIBUTES_TO_CLONE = (
    "states",
    "observations",
    "rewards",
    "dones",
    "scores",
    "observations",
)


class FMC:
    def __init__(
        self, 
        vectorized_environment: VectorizedEnvironment,
        balance: float = 1.0,
        similarity_function: Callable = _l2_distance,
        freeze_best: bool = True,
        track_tree: bool = True,
    ):
        self.vec_env = vectorized_environment
        self.balance = balance
        self.similarity_function = similarity_function

        self.freeze_best = freeze_best
        self.track_tree = track_tree

        self.reset()

    def reset(self):
        self.vec_env.batch_reset()
        self.states, self.observations, self.rewards, self.dones, self.infos = None, None, None, None, None

        self.scores = torch.zeros(self.num_walkers, dtype=float)
        self.clone_mask = torch.zeros(self.num_walkers, dtype=bool)
        self.freeze_mask = torch.zeros((self.num_walkers), dtype=bool)

        self.tree = GameTree(self.num_walkers, prune=True) if self.track_tree else None

    @property
    def num_walkers(self):
        return self.vec_env.n

    @torch.no_grad()
    def simulate(self, steps: int, use_tqdm: bool = False):
        it = tqdm(range(steps), disable=not use_tqdm)
        for _ in it:
            self._perturbate()
            self._clone()

    def _perturbate(self):
        self.actions = self.vec_env.batched_action_space_sample()
        self.states, self.observations, self.rewards, self.dones, self.infos = self.vec_env.batch_step(self.actions, self.freeze_mask)
        self.scores += self.rewards

        if self.tree:
            self.tree.build_next_level(self.actions, self.observations, self.rewards, self.freeze_mask)
        self._set_freeze_mask()

    def _set_freeze_mask(self):
        self.freeze_mask = torch.zeros((self.num_walkers), dtype=bool)
        if self.freeze_best:
            self.freeze_mask[self.scores.argmax()] = 1

    def _set_clone_variables(self):
        self.clone_partners = torch.randperm(self.num_walkers)
        self.similarities = self.similarity_function(self.states, self.states[self.clone_partners])

        rel_sim = _relativize_vector(self.similarities)
        rel_score = _relativize_vector(self.scores)
        self.virtual_rewards = rel_score ** self.balance * rel_sim

        vr = self.virtual_rewards
        pair_vr = self.virtual_rewards[self.clone_partners]
        value = (pair_vr - vr) / torch.where(vr > 0, vr, 1e-8)
        self.clone_mask = (value >= torch.rand(1)).bool()

        # don't clone frozen walkers
        self.clone_mask[self.freeze_mask] = False

        # TODO: include `self.dones` into decision

    def _clone(self):
        self._set_clone_variables()

        self.vec_env.clone(self.clone_partners, self.clone_mask)
        if self.tree:
            self.tree.clone(self.clone_partners, self.clone_mask)

        for attr in _ATTRIBUTES_TO_CLONE:
            self._clone_variable(attr)

    def _clone_vector_inplace(self, vector):
        vector[self.clone_mask] = vector[self.clone_partners[self.clone_mask]]
        return vector

    def _clone_list(self, l: List, copy: bool = False):
        new_list = []
        for i in range(self.num_walkers):
            do_clone = self.clone_mask[i]
            partner = self.clone_partners[i]

            if do_clone:
                # NOTE: may not need to deepcopy.
                if copy:
                    new_list.append(deepcopy(l[partner]))
                else:
                    new_list.append(l[partner])
            else:
                new_list.append(l[i])
        return new_list

    def _clone_variable(self, subject):
        if isinstance(subject, torch.Tensor):
            return self._clone_vector_inplace(subject)
        elif isinstance(subject, list):
            return self._clone_list(subject)
        elif isinstance(subject, str):
            cloned_subject = self._clone_variable(getattr(self, subject))
            setattr(self, subject, cloned_subject)
            return cloned_subject
        raise NotImplementedError()