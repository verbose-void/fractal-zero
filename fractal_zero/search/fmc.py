from typing import Callable
import torch
import numpy as np

from tqdm import tqdm
from fractal_zero.search.tree import GameTree
from fractal_zero.utils import cloning_primitive, normalize_and_log_exp

from fractal_zero.vectorized_environment import VectorizedEnvironment

_ATTRIBUTES_TO_CLONE = (
    "states",
    "observations",
    "rewards",
    "dones",
    "scores",
    "average_scores",
    "actions",
    "infos",
)

class FMC:
    def __init__(
        self,
        vectorized_environment: VectorizedEnvironment,
        balance: float = 1.0,
        disable_cloning: bool = False,
        use_average_rewards: bool = False,
        similarity_function: Callable = torch.dist,
        freeze_best: bool = True,
        track_tree: bool = True,
        prune_tree: bool = True,
    ):
        self.vec_env = vectorized_environment
        self.balance = balance
        self.disable_cloning = disable_cloning
        self.use_average_rewards = use_average_rewards
        self.similarity_function = similarity_function

        self.freeze_best = freeze_best
        self.track_tree = track_tree
        self.prune_tree = prune_tree

        self.reset()
    
    @property
    def num_walkers(self):
        return self.vec_env.n

    def reset(self):
        # TODO: may need to make this decision of root observations more effectively for stochastic environments.
        self.observations = self.vec_env.batch_reset()
        root_observation = self.observations[0]

        self.dones = torch.zeros(self.num_walkers).bool()
        self.states, self.observations, self.rewards, self.infos = (
            None,
            None,
            None,
            None,
        )

        self.scores = torch.zeros(self.num_walkers, dtype=float)
        self.average_scores = torch.zeros(self.num_walkers, dtype=float)
        self.clone_mask = torch.zeros(self.num_walkers, dtype=bool)
        self.freeze_mask = torch.zeros((self.num_walkers), dtype=bool)

        self.tree = (
            GameTree(self.num_walkers, prune=self.prune_tree, root_observation=root_observation)
            if self.track_tree
            else None
        )
        self.did_early_exit = False

    def simulate(self, steps: int, use_tqdm: bool = False):
        if self.did_early_exit:
            raise ValueError("Already early exited.")

        for _ in tqdm(range(steps), disable=not use_tqdm):
            self._perturbate()

            if self._can_early_exit():
                self.did_early_exit = True
                break
            
            self._clone_walkers()

    def _perturbate(self):
        """
        Perturbate the walkers by sampling actions from the action space and
        stepping the environment.
        """
        # Why logical or here? Do you now always want for freeze if the walker is in the terminal state?
        # What if done is 1 and freeze_mask is 1?
        freeze_steps = torch.logical_or(self.freeze_mask, self.dones)

        # TODO: don't sample actions for frozen environments? (make sure to remove the comments about this)
        # will make it more legible.
        self.actions = self.vec_env.batched_action_space_sample()

        (
            self.states,
            self.observations,
            self.rewards,
            self.dones,
            self.infos,
        ) = self.vec_env.batch_step(self.actions, freeze_steps)
        self.scores += self.rewards
        self.average_scores = self.scores / self.tree.get_depths()

        if self.tree:
            # NOTE: the actions that are in the tree will diverge slightly from
            # those being represented by FMC's internal state. The reason for this is
            # that when we freeze certain environments in the vectorized environment
            # object, the actions that are sampled will not be enacted, and the previous
            # return values will be provided.
            self.tree.build_next_level(
                self.actions,
                self.observations,
                self.rewards,
                self.infos,
                freeze_steps,
            )

        self._set_freeze_mask()
    
    def _clone_walkers(self):
        self._set_clone_variables()

        if self.disable_cloning:
            return
        self.vec_env.clone(self.clone_partners, self.clone_mask)
        if self.tree:
            self.tree.clone(self.clone_partners, self.clone_mask)

        # doing this allows the GameTree to retain gradients in case training a model on FMC outputs.
        # it also is required to keep the cloning mechanism in-tact (because of inplace updates).
        self.rewards = self.rewards.clone()
        if isinstance(self.observations, torch.Tensor):
            self.observations = self.observations.clone()
        if isinstance(self.infos, torch.Tensor):
            self.infos = self.infos.clone()

        # What is this for? 
        for attr in _ATTRIBUTES_TO_CLONE:
            self._clone_variable(attr)

        # sanity checks (TODO: maybe remove this?)
        if not torch.allclose(self.scores, self.tree.get_total_rewards(), rtol=0.001):
            raise ValueError(self.scores, self.tree.get_total_rewards())
        # if self.rewards[self.freeze_mask].sum().item() != 0:
        #     raise ValueError(self.rewards[self.freeze_mask], self.rewards[self.freeze_mask].sum())
    
    def _set_freeze_mask(self):
        self.freeze_mask = torch.zeros((self.num_walkers), dtype=bool)
        if self.freeze_best:
            self.freeze_mask[self._score_walkers().argmax()] = 1
    
    def _set_clone_variables(self):
        self._set_valid_clone_partners()
        self._set_clone_mask()
    

    def _set_valid_clone_partners(self):
        # cannot clone to walkers at terminal states
        valid_clone_partners = np.arange(self.num_walkers)[
            (self.dones == False).numpy()
        ]

        # TODO: make it so walkers cannot clone to themselves
        clone_partners = np.random.choice(valid_clone_partners, size=self.num_walkers)
        self.clone_partners = torch.tensor(clone_partners, dtype=int).long()
    
    def _set_clone_mask(self):
        values = self._get_walker_values()
        self.clone_mask: torch.Tensor[bool] = (values >= torch.rand(1)).bool()

        # clone all walkers at terminal states and not frozen walkers
        self.clone_mask[
            self.dones
        ] = True  # NOTE: sometimes done might be a preferable terminal state (winning)... deal with this.
        self.clone_mask[self.freeze_mask] = False
    
    def _get_walker_values(self) -> torch.Tensor:
        """
        Walker values are scored and used to determine which walkers should be cloned.
        """
        walker_partner_similarities: torch.Tensor = self.similarity_function(
            self.states, self.states[self.clone_partners]
        )

        relativized_walker_similarity_scores: torch.Tensor = normalize_and_log_exp(
            walker_partner_similarities
        )
        relativized_walker_scores: torch.Tensor = (
            normalize_and_log_exp(self.average_scores)
            if self.use_average_rewards
            else normalize_and_log_exp(self.scores)
        )

        # Why do we power the walker exploitation scores?
        virtual_rewards: torch.Tensor = (
            relativized_walker_scores**self.balance
            * relativized_walker_similarity_scores
        )

        pair_virtual_rewards = virtual_rewards[self.clone_partners]

        # What is `value`?
        return (pair_virtual_rewards - virtual_rewards) / torch.where(
            virtual_rewards > 0, virtual_rewards, 1e-8
        )

    def _score_walkers(self) -> torch.Tensor:
        return self.average_scores if self.use_average_rewards else self.scores
    
    def _can_early_exit(self) -> torch.Tensor:
        return torch.all(self.dones)

    def _clone_variable(self, subject_var_name: str):
        subject = getattr(self, subject_var_name)
        # note: this will be cloned in-place!
        cloned_subject = cloning_primitive(
            subject, self.clone_partners, self.clone_mask
        )
        setattr(self, subject_var_name, cloned_subject)
        return cloned_subject
