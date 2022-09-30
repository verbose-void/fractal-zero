from copy import deepcopy
from typing import Callable, Dict, Union
import torch
import torch.nn.functional as F
import gym
import wandb

from fractal_zero.loss.space_loss import get_space_loss
from fractal_zero.search.fmc import FMC
from fractal_zero.config import FMCConfig
from fractal_zero.utils import (
    dist_of_model_paramters,
    parameters_norm,
)

from fractal_zero.vectorized_environment import (
    RayVectorizedEnvironment,
    SerialVectorizedEnvironment,
    load_environment,
)


class OnlineFMCPolicyTrainer:
    """Trains a policy model in an online manner using FMC as the data generator. "Online" means that the latest policy
    weights are used during the search process. So the data is generated, the model is trained, and the cycle continues.
    """

    def __init__(
        self,
        env: Union[str, gym.Env],
        policy_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        num_walkers: int,
        observation_encoder: Callable = None,
        loss_spec = None,
        fmc_config: FMCConfig = None,
        use_ray: bool = False,
    ):
        self.env = load_environment(env)

        # TODO: config option
        if use_ray:
            self.vec_env = RayVectorizedEnvironment(env, num_walkers, observation_encoder=observation_encoder)
        else:
            self.vec_env = SerialVectorizedEnvironment(env, num_walkers, observation_encoder=observation_encoder)

        self.policy_model = policy_model
        self.optimizer = optimizer
        self.action_loss = get_space_loss(self.env.action_space, spec=loss_spec)

        self.most_reward = float("-inf")
        self.best_model = None

        self.fmc_config = fmc_config

    def generate_episode_data(self, max_steps: int):
        self.vec_env.batch_reset()

        self.fmc = FMC(self.vec_env, config=self.fmc_config)  # , policy_model=self.policy_model)
        self.fmc.simulate(max_steps)

    def _get_best_only_batch(self):
        observations = []
        actions = []

        path = self.fmc.tree.best_path
        for state, action in path:
            # obs = torch.tensor(state.observation)
            observations.append(state.observation)
            actions.append([action])

        # x = torch.stack(observations).float()
        # t = torch.tensor(actions)
        # return [x], [t]

        # TODO: instead of using weights, use the visit counts!
        weights = torch.ones((1, 1, len(observations))).float()

        return observations, actions, weights

    def _get_batch(self):
        if not self.fmc.tree:
            raise ValueError("FMC is not tracking walker paths.")

        # TODO: config
        best_only = False

        if best_only:
            return self._get_best_only_batch()

        # convert the game tree into a weighted set of targets, based on the visit counts

        observations = []
        child_actions = []
        child_weights = []

        g = self.fmc.tree.g
        for node in g.nodes:
            observations.append(node.observation)

            actions = []
            weights = []
            for _, child_node, data in g.out_edges(node, data=True):
                # TODO: make this measure based on a proprotion of total clone receives?
                weight = child_node.num_child_walkers / self.fmc.num_walkers

                weights.append(weight)

                action = data["action"]
                actions.append(action)

            child_actions.append(actions)
            child_weights.append(torch.tensor(weights).float())

        return observations, child_actions, child_weights

    def _general_loss(self, action, action_targets, action_weights):
        # TODO: vectorize better (if possible?)
        loss = 0
        for action_target, weight in zip(action_targets, action_weights):
            loss += self.action_loss(action, action_target) * weight
        return loss

    def train_on_latest_episode(self):
        self.policy_model.train()
        self.optimizer.zero_grad()

        # TODO: make more efficient somehow?
        self.params_before = deepcopy(list(self.policy_model.parameters()))

        observations, actions, weights = self._get_batch()
        # assert len(observations) == len(actions) == len(weights)
        assert len(observations) == len(actions)

        # NOTE: loss for trajectories of weighted multi-target actions
        loss = 0
        action_predictions = self.policy_model.forward(observations)
        for y, action_targets, action_weights in zip(
            action_predictions, actions, weights,
        ):
            trajectory_loss = self._general_loss(
                y, action_targets, action_weights
            )
            loss += trajectory_loss
        loss = loss / len(observations)

        loss.backward()
        self.optimizer.step()

        self._log_last_train_step(loss.item())
        return loss.item()

    def evaluate_policy(self, max_steps: int, render: bool = False, evaluate_best_policy: bool = False):
        policy = self.best_model if evaluate_best_policy else self.policy_model

        policy.eval()

        obs = self.env.reset()

        rewards = []

        for _ in range(max_steps):
            action = policy.forward(obs)
            action = policy.parse_action(action)
            obs, reward, done, info = self.env.step(action)
            rewards.append(reward)

            if render:
                self.env.render()

            if done:
                break

        if not evaluate_best_policy:
            total_reward = sum(rewards)
            if total_reward > self.most_reward:
                self.most_reward = total_reward
                self.best_model = deepcopy(self.policy_model)

        self._log_last_eval_step(rewards)
        return sum(rewards)

    def _log_last_train_step(self, train_loss: float):
        if wandb.run is None:
            return

        best_path = self.fmc.tree.best_path
        last_episode_total_reward = best_path.total_reward

        current_params = list(self.policy_model.parameters())
        param_dist = dist_of_model_paramters(self.params_before, current_params)
        param_norm = parameters_norm(current_params)

        wandb.log(
            {
                "train/loss": train_loss,
                "train/epsiode_reward": last_episode_total_reward,
                "parameters/policy_norm": param_norm,
                "parameters/policy_l2_distance": param_dist,
            }
        )

    def _log_last_eval_step(self, rewards):
        if wandb.run is None:
            return

        wandb.log(
            {
                "eval/total_rewards": sum(rewards),
            }
        )

    def replay_best(self, render: bool=False):
        _, actions = self._get_best_only_batch()
        self.env.reset()

        for action in actions:
            self.env.step(action)

            if render:
                self.env.render()