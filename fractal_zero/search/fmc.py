from copy import deepcopy
from typing import Callable, List
import torch
import numpy as np

from tqdm import tqdm
from fractal_zero.search.tree import GameTree

from fractal_zero.vectorized_environment import VectorizedEnvironment


def _l2_distance(vec0, vec1):
    if vec0.dim() > 2:
        vec0 = vec0.flatten(start_dim=1)
    if vec1.dim() > 2:
        vec1 = vec1.flatten(start_dim=1)
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
        # TODO: may need to make this decision of root observations more effectively for stochastic environments.
        self.observations = self.vec_env.batch_reset()
        root_obs = self.observations[0]

        self.dones = torch.zeros(self.num_walkers).bool()
        self.states, self.observations, self.rewards, self.infos = (
            None,
            None,
            None,
            None,
        )

        self.scores = torch.zeros(self.num_walkers, dtype=float)
        self.clone_mask = torch.zeros(self.num_walkers, dtype=bool)
        self.freeze_mask = torch.zeros((self.num_walkers), dtype=bool)

        self.tree = GameTree(self.num_walkers, prune=True, root_observation=root_obs) if self.track_tree else None

    @property
    def num_walkers(self):
        return self.vec_env.n

    def _can_early_exit(self):
        return torch.all(self.dones)

    @torch.no_grad()
    def simulate(self, steps: int, use_tqdm: bool = False):
        it = tqdm(range(steps), disable=not use_tqdm)
        for _ in it:
            self._perturbate()

            if self._can_early_exit():
                break

            self._clone()

    def _perturbate(self):
        self.actions = self.vec_env.batched_action_space_sample()

        freeze_steps = torch.logical_or(self.freeze_mask, self.dones)
        (
            self.states,
            self.observations,
            self.rewards,
            self.dones,
            self.infos,
        ) = self.vec_env.batch_step(self.actions, freeze_steps)
        self.scores += self.rewards

        if self.tree:
            self.tree.build_next_level(
                self.actions, self.observations, self.rewards, freeze_steps,
            )
        self._set_freeze_mask()

    def _set_freeze_mask(self):
        self.freeze_mask = torch.zeros((self.num_walkers), dtype=bool)
        if self.freeze_best:
            self.freeze_mask[self.scores.argmax()] = 1

    def _set_valid_clone_partners(self):
        valid_clone_partners = np.arange(self.num_walkers)

        # cannot clone to walkers at terminal states
        valid_clone_partners = valid_clone_partners[(self.dones == False).numpy()]

        # TODO: make it so walkers cannot clone to themselves
        clone_partners = np.random.choice(valid_clone_partners, size=self.num_walkers)
        self.clone_partners = torch.tensor(clone_partners, dtype=int).long()

    def _set_clone_variables(self):
        self._set_valid_clone_partners()
        self.similarities = self.similarity_function(
            self.states, self.states[self.clone_partners]
        )

        rel_sim = _relativize_vector(self.similarities)
        rel_score = _relativize_vector(self.scores)
        self.virtual_rewards = rel_score**self.balance * rel_sim

        vr = self.virtual_rewards
        pair_vr = self.virtual_rewards[self.clone_partners]
        value = (pair_vr - vr) / torch.where(vr > 0, vr, 1e-8)
        self.clone_mask = (value >= torch.rand(1)).bool()

        # clone all walkers at terminal states
        self.clone_mask[self.dones] = True
        # don't clone frozen walkers
        self.clone_mask[self.freeze_mask] = False

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
