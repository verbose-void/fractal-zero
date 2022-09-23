from warnings import warn
import torch
import numpy as np
from tqdm import tqdm

import wandb

from fractal_zero.config import FMCConfig, FractalZeroConfig

from fractal_zero.models.joint_model import JointModel
from fractal_zero.utils import mean_min_max_dict
from fractal_zero.vectorized_environment import (
    VectorizedDynamicsModelEnvironment,
    VectorizedEnvironment,
)


@torch.no_grad()
def _relativize_vector(vector):
    std = vector.std()
    if std == 0:
        return torch.ones(len(vector))
    standard = (vector - vector.mean()) / std
    standard[standard > 0] = torch.log(1 + standard[standard > 0]) + 1
    standard[standard <= 0] = torch.exp(standard[standard <= 0])
    return standard


class FMC:
    """Fractal Monte Carlo is a collaborative cellular automata based tree search algorithm. This version is special, because instead of having a gym
    environment maintain the state for each walker during the search process, each walker's state is represented inside of a batched hidden
    state variable inside of a dynamics model. Basically, the dynamics model's hidden state is of shape (num_walkers, *embedding_shape).

    This is inspired by Muzero's technique to have a dynamics model be learned such that the tree search need not interact with the environment
    itself. With FMC, it is much more natural than with MCTS, mostly because of the cloning phase being contrastive. As an added benefit of this
    approach, it's natively vectorized so it can be put onto the GPU.
    """

    vectorized_environment: VectorizedEnvironment

    def __init__(
        self,
        vectorized_environment: VectorizedEnvironment,
        prediction_model: torch.nn.Module = None,
        config: FMCConfig = None,
        verbose: bool = False,
    ):
        self.vectorized_environment = vectorized_environment
        self.model = prediction_model
        self.verbose = verbose

        if config is None:
            self.config = self._build_default_config()
        else:
            self.config = config
        self._validate_config()

    def _build_default_config(self) -> FMCConfig:
        use_actual_env = not isinstance(
            self.vectorized_environment, VectorizedDynamicsModelEnvironment
        )

        return FMCConfig(
            num_walkers=self.vectorized_environment.n,
            search_using_actual_environment=use_actual_env,
            clone_strategy="predicted_values"
            if self.model is not None
            else "cumulative_reward",
        )

    def _validate_config(self):
        if self.config.num_walkers != self.vectorized_environment.n:
            raise ValueError(
                f"Expected config num walkers ({self.config.num_walkers}) and vectorized environment n ({self.vectorized_environment.n}) to match."
            )

        if self.model is None and self.config.clone_strategy == "predicted_values":
            raise ValueError(
                "Cannot clone based on predicted values when no model is provided. Change the strategy or provide a policy + value model."
            )

    @property
    def num_walkers(self) -> int:
        return self.config.num_walkers

    @property
    def device(self):
        return self.config.device

    @property
    def batch_actions(self):
        if self.config.search_using_actual_environment:
            return self.raw_actions
        return self.actions

    @torch.no_grad()
    def _perturbate(self):
        """Advance the state of each walker."""

        self._assign_actions()

        self.observations, self.rewards, self.dones, _ = self.vectorized_environment.batch_step(
            self.batch_actions
        )

        if self.model is not None:
            _, self.predicted_values = self.model.forward(self.observations)

        self.reward_buffer[:, self.simulation_iteration] = self.rewards

    @torch.no_grad()
    def simulate(self, k: int, greedy_action: bool = True, use_tqdm: bool = False):
        """Run FMC for k iterations, returning the best action that was taken at the root/initial state."""

        self.k = k
        assert self.k > 0

        # TODO: explain all these variables
        # NOTE: they should exist on the CPU.
        self.reward_buffer = torch.zeros(
            size=(self.num_walkers, self.k, 1),
            dtype=float,
        )
        self.value_sum_buffer = torch.zeros(
            size=(self.num_walkers, 1),
            dtype=float,
        )
        self.visit_buffer = torch.zeros(
            size=(self.num_walkers, 1),
            dtype=int,
        )
        self.clone_receives = torch.zeros(
            size=(self.num_walkers, 1),
            dtype=int,
        )

        self.root_actions = None
        self.root_value_sum = 0
        self.root_visits = 0

        it = tqdm(range(self.k), desc="Simulating with FMC", total=self.k, disable=not use_tqdm)
        for self.simulation_iteration in it:
            self._perturbate()
            self._prepare_clone_variables()
            self._backpropagate_reward_buffer()
            self._execute_cloning()

        # TODO: try to convert the root action distribution into a policy distribution? this may get hard in continuous action spaces. https://arxiv.org/pdf/1805.09613.pdf

        self.log(
            {
                **mean_min_max_dict("fmc/visit_buffer", self.visit_buffer.float()),
                **mean_min_max_dict("fmc/value_sum_buffer", self.value_sum_buffer),
                **mean_min_max_dict(
                    "fmc/average_value_buffer",
                    self.value_sum_buffer / self.visit_buffer.float(),
                ),
                **mean_min_max_dict("fmc/clone_receives", self.clone_receives.float()),
            },
            commit=False,
        )

        # TODO: experiment with these
        if greedy_action:
            return self._get_highest_value_action()
        return self._get_action_with_highest_clone_receives()

    @torch.no_grad()
    def _assign_actions(self):
        """Each walker picks an action to advance it's state."""

        # TODO: use the policy function for action selection.
        self.raw_actions = self.vectorized_environment.batched_action_space_sample()
        self.actions = torch.tensor(self.raw_actions, device=self.device).unsqueeze(-1)

        if self.root_actions is None:
            self.root_actions = self.actions.cpu().detach().clone()

    @torch.no_grad()
    def _assign_clone_partners(self):
        """For the cloning phase, walkers need a partner to determine if they should be sent as reinforcements to their partner's state."""

        choices = np.random.choice(np.arange(self.num_walkers), size=self.num_walkers)
        self.clone_partners = torch.tensor(choices, dtype=int)

    @torch.no_grad()
    def _calculate_distances(self):
        """For the cloning phase, we calculate the distances between each walker and their partner for balancing exploration."""

        self.distances = torch.linalg.norm(
            self.observations - self.observations[self.clone_partners], dim=1
        )

    @torch.no_grad()
    def _calculate_virtual_rewards(self):
        """For the cloning phase, we calculate a virtual reward that is the composite of each walker's distance to their partner weighted with
        their rewards. This is used to determine the probability to clone and is used to balance exploration and exploitation.

        Both the reward and distance vectors are "relativized". This keeps all of the values in each vector contextually scaled with each step.
        The authors of Fractal Monte Carlo claim this is a method of shaping a "universal reward function". Without relativization, the
        vectors may have drastically different ranges, causing more volatility in how many walkers are cloned at each step. If the reward or distance
        ranges were too high, it's likely no cloning would occur at all. If either were too small, then it's likely all walkers would be cloned.
        """

        # TODO EXPERIMENT: should we be using the value estimates? or should we be using the value buffer?
        # or should we be using the cumulative rewards? (the original FMC authors use cumulative rewards)
        clone_strat = self.config.clone_strategy
        if clone_strat == "predicted_values":
            exploit = self.predicted_values
        elif clone_strat == "cumulative_reward":
            # TODO cache this...
            cumulative_rewards = self.reward_buffer[:, : self.simulation_iteration].sum(
                dim=1
            )
            exploit = cumulative_rewards

        rel_exploits = _relativize_vector(exploit).squeeze(-1).cpu()
        rel_distances = _relativize_vector(self.distances).cpu()
        self.virtual_rewards = (rel_exploits**self.config.balance) * rel_distances

        self.log(
            {
                **mean_min_max_dict("fmc/virtual_rewards", self.virtual_rewards),
                # **mean_min_max_dict("fmc/predicted_values", self.predicted_values),
                **mean_min_max_dict("fmc/distances", self.distances),
                **mean_min_max_dict("fmc/auxiliaries", self.rewards),
            },
            commit=False,
        )

    @torch.no_grad()
    def _determine_clone_receives(self):
        # keep track of which walkers received clones and how many.
        clones_received_per_walker = torch.bincount(
            self.clone_partners[self.clone_mask]
        ).unsqueeze(-1)

        n = len(clones_received_per_walker)

        self.clone_receives[:n] += clones_received_per_walker

        self.clone_receive_mask = torch.zeros_like(self.clone_mask)
        self.clone_receive_mask[:n] = clones_received_per_walker.squeeze(-1) > 0

    @torch.no_grad()
    def _determine_clone_mask(self):
        """The clone mask is based on the virtual rewards of each walker and their clone partner. If a walker is selected to clone, their
        state will be replaced with their partner's state.
        """

        vr = self.virtual_rewards
        pair_vr = vr[self.clone_partners]

        self.clone_probabilities = (pair_vr - vr) / torch.where(vr > 0, vr, 1e-8)
        r = np.random.uniform()
        self.clone_mask = (self.clone_probabilities >= r).cpu()

        self._determine_clone_receives()

        self.log(
            {
                "fmc/num_cloned": self.clone_mask.sum(),
            },
            commit=False,
        )

    @torch.no_grad()
    def _prepare_clone_variables(self):
        # TODO: docstring

        # prepare virtual rewards and partner virtual rewards
        self._assign_clone_partners()
        self._calculate_distances()
        self._calculate_virtual_rewards()
        self._determine_clone_mask()

    @torch.no_grad()
    def _execute_cloning(self):
        """The cloning phase is where the collaboration of the cellular automata comes from. Using the virtual rewards calculated for
        each walker and clone partners that are randomly assigned, there is a probability that some walkers will be sent as reinforcements
        to their randomly assigned clone partner.

        The goal of the clone phase is to maintain a balanced density over state occupations with respect to exploration and exploitation.
        """

        # TODO: don't clone best walker (?)
        # execute clones
        # self._clone_vector(self.state)
        self.vectorized_environment.clone(self.clone_partners, self.clone_mask)

        self._clone_vector(self.observations)
        self._clone_vector(self.rewards)

        self._clone_vector(self.actions)
        self._clone_vector(self.root_actions)
        self._clone_vector(self.reward_buffer)
        self._clone_vector(self.value_sum_buffer)
        self._clone_vector(self.visit_buffer)
        self._clone_vector(self.clone_receives)  # yes... clone clone receives lol.

        if self.verbose:
            print("state after", self.state)

    def _get_backprop_mask(self):
        # TODO: docstring

        # usually, we only backpropagate the walkers who are about to clone away. However, at the very end of the simulation, we want
        # to backpropagate the value regardless of if they are cloning or not.
        # TODO: experiment with this, i'm not sure if it's better to always backpropagate all or only at the end. it's an open question.
        force_backpropagate_all = self.simulation_iteration == self.k - 1

        strat = self.config.backprop_strategy
        if strat == "all" or force_backpropagate_all:
            mask = torch.ones_like(self.clone_mask)
        elif strat == "clone_mask":
            mask = self.clone_mask
        elif strat == "clone_participants":
            mask = torch.logical_or(self.clone_mask, self.clone_receive_mask)
        else:
            raise ValueError(f'FMC Backprop strategy "{strat}" is not supported.')

        return mask

    def _backpropagate_reward_buffer(self):
        """This essentially does the backpropagate step that MCTS does, although instead of maintaining an entire tree, it maintains
        value sums and visit counts for each walker. These values may be subsequently cloned. There is some information loss
        during this clone, but it should be minimally impactful.
        """

        mask = self._get_backprop_mask()

        current_value_buffer = torch.zeros_like(self.value_sum_buffer)
        for i in reversed(range(self.simulation_iteration)):
            current_value_buffer[mask] = (
                self.reward_buffer[mask, i]
                + current_value_buffer[mask] * self.config.gamma
            )

        self.value_sum_buffer += current_value_buffer
        self.visit_buffer += mask.unsqueeze(-1)

        self.root_value_sum += current_value_buffer.sum()
        self.root_visits += mask.sum()

    @property
    def root_value(self):
        """Kind of equivalent to the MCTS root value."""

        return (self.root_value_sum / self.root_visits).item()

    @torch.no_grad()
    def _get_highest_value_action(self):
        """The highest value action corresponds to the walker whom has the highest average estimated value."""

        self.walker_values = self.value_sum_buffer / self.visit_buffer
        highest_value_walker_index = torch.argmax(self.walker_values)
        highest_value_action = (
            self.root_actions[highest_value_walker_index, 0].cpu().numpy()
        )

        return highest_value_action

    @torch.no_grad()
    def _get_action_with_highest_clone_receives(self):
        # TODO: docstring
        most_cloned_to_walker = torch.argmax(self.clone_receives)
        return self.root_actions[most_cloned_to_walker, 0].cpu().numpy()

    def _clone_vector(self, vector: torch.Tensor):
        vector[self.clone_mask] = vector[self.clone_partners[self.clone_mask]]

    def log(self, *args, **kwargs):
        # TODO: separate logger class

        if not self.config.use_wandb:
            return

        if wandb.run is None:
            warn(
                "Weights and biases config was provided, but wandb.init was not called."
            )
            return

        wandb.log(*args, **kwargs)
