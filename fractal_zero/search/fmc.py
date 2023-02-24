from typing import Callable, List, Optional, Tuple, Type
import torch
import numpy as np

from tqdm import tqdm
from fractal_zero.search.tree import GameTree, StepOutputs
from fractal_zero.utils import cloning_primitive

from fractal_zero.vectorized_environment import VectorizedEnvironment

def normalize_and_log_exp(vector: torch.Tensor) -> torch.Tensor:
    mean = torch.mean(vector)
    std = torch.std(vector)
    if std == 0:
        return torch.ones(len(vector))
    
    # Subtract the mean from the vector, and divide by the standard deviation.
    # This "standardizes" the vector so that it has mean 0 and standard deviation 1.
    relativized_vector: torch.Tensor = (vector - mean) / std

    # Apply a non-linear transformation to the standardized vector that makes it more sensitive to differences in the tails of the distribution.
    # All values are positive, because the `balance` hyperparameter exponentiates the vector, and want to maintain monotonicity.
    return torch.where(relativized_vector > 0, torch.log1p(relativized_vector) + 1, torch.exp(relativized_vector))


_ATTRIBUTES_TO_CLONE = (
    "states",
    "observations",
    "rewards",
    "dones",
    "scores",
    "average_rewards",
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

    def reset(self):
        # TODO: may need to make this decision of root observations more effectively for stochastic environments.
        self.observations = self.vec_env.batch_reset()
        root_obs = self.observations[0]

        self.tree = (
            GameTree(self.num_walkers, prune=self.prune_tree, root_observation=root_obs)
            if self.track_tree
            else None
        )
        self.did_early_exit = False

    @property
    def num_walkers(self):
        return self.vec_env.n

    def simulate(self, steps: int, use_tqdm: bool = False):
        if self.did_early_exit:
            raise ValueError("Already early exited.")
        dones = torch.zeros(self.num_walkers)
        rewards = torch.zeros(self.num_walkers)
        average_rewards = torch.zeros(self.num_walkers)
        for _ in tqdm(range(steps), disable=not use_tqdm):
            # Determine which walkers to freeze - not perturbed
            freeze_mask = self._get_freeze_mask()
            freeze_steps = torch.logical_or(freeze_mask, dones)
            actions = self.vec_env.batched_action_space_sample()
            step_outputs = StepOutputs(self.vec_env.batch_step(actions, freeze_steps))

            # track simulated rewards for each walker
            rewards += step_outputs.rewards
            average_rewards = rewards / self.tree.get_depths()

            if self.tree:
                # NOTE: the actions that are in the tree will diverge slightly from
                # those being represented by FMC's internal state. The reason for this is
                # that when we freeze certain environments in the vectorized environment
                # object, the actions that are sampled will not be enacted, and the previous
                # return values will be provided.
                self.tree.build_next_level(
                    actions,
                    step_outputs.observations,
                    step_outputs.rewards,
                    step_outputs.infos,
                    freeze_steps,
                )

            # Break if all walkers in a terminal state
            if self._can_early_exit(dones):
                self.did_early_exit = True
                break

            if self.disable_cloning:
                return

            ## Get clone partners not in terminal state
            valid_clone_partners = np.arange(self.num_walkers)[(dones == False).numpy()]

            # TODO: make it so walkers cannot clone to themselves
            clone_partners = np.random.choice(valid_clone_partners, size=self.num_walkers)
            clone_partners = torch.tensor(clone_partners, dtype=int).long()

            ## Calculate walker similarities and relativized rewards
            walker_partner_similarities = self.similarity_function(
                step_outputs.states, step_outputs.states[clone_partners]
            )

            rel_walker_similarity = normalize_and_log_exp(walker_partner_similarities)
            rel_walker_score = normalize_and_log_exp(average_rewards) if self.use_average_rewards else normalize_and_log_exp(rewards)

            virtual_rewards = rel_walker_score**self.balance * rel_walker_similarity

            pair_vr = virtual_rewards[clone_partners]
            value = (pair_vr - virtual_rewards) / torch.where(virtual_rewards > 0, virtual_rewards, 1e-8)
            
            clone_mask = (value >= torch.rand(1)).bool()

            # clone all walkers at terminal states and don't clone walkers that are frozen
            clone_mask[dones] = True  # NOTE: sometimes done might be a preferable terminal state (winning)... deal with this.
            clone_mask[freeze_mask] = False

            self.vec_env.clone(clone_partners, clone_mask)
            if self.tree:
                self.tree.clone(clone_partners, clone_mask)

            # doing this allows the GameTree to retain gradients in case training a model on FMC outputs.
            # it also is required to keep the cloning mechanism in-tact (because of inplace updates).
            step_outputs.rewards = step_outputs.rewards.clone()
            if isinstance(step_outputs.observations, torch.Tensor):
                step_outputs.observations = step_outputs.observations.clone()
            if isinstance(step_outputs.infos, torch.Tensor):
                step_outputs.infos = step_outputs.infos.clone()

            for attr in _ATTRIBUTES_TO_CLONE:
                self._clone_variable(attr)

            # sanity checks (TODO: maybe remove this?)
            if not torch.allclose(rewards, self.tree.get_total_rewards(), rtol=0.001):
                raise ValueError(rewards, self.tree.get_total_rewards())
            # if self.rewards[self.freeze_mask].sum().item() != 0:
            #     raise ValueError(self.rewards[self.freeze_mask], self.rewards[self.freeze_mask].sum())

    @staticmethod
    def _can_early_exit(dones) -> torch.Tensor:
        return torch.all(dones)
    
    def _get_freeze_mask(self, scores, average_rewards):
        freeze_mask = torch.zeros((self.num_walkers), dtype=bool)
        if self.freeze_best:
            metric = average_rewards if self.use_average_rewards else scores
            freeze_mask[metric.argmax()] = 1
        return freeze_mask        

    def _clone_variable(self, subject_var_name: str, clone_mask: torch.Tensor, clone_partners: torch.Tensor):
        subject = getattr(self, subject_var_name)
        # note: this will be cloned in-place!
        cloned_subject = cloning_primitive(
            subject, clone_partners, clone_mask
        )
        setattr(self, subject_var_name, cloned_subject)
        return cloned_subject
