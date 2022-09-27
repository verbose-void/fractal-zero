from typing import Union
import torch
import torch.nn.functional as F
import gym
import wandb

from fractal_zero.data.replay_buffer import GameHistory
from fractal_zero.search.fmc import FMC

from fractal_zero.vectorized_environment import RayVectorizedEnvironment, load_environment


class OnlineFMCPolicyTrainer:
    """Trains a policy model in an online manner using FMC as the data generator. "Online" means that the latest policy
    weights are used during the search process. So the data is generated, the model is trained, and the cycle continues.
    """

    def __init__(self, env: Union[str, gym.Env], policy_model: torch.nn.Module, optimizer: torch.optim.Optimizer, num_walkers: int):
        self.env = load_environment(env)
        self.vec_env = RayVectorizedEnvironment(env, num_walkers)

        self.policy_model = policy_model
        self.optimizer = optimizer

    def generate_episode_data(self, max_steps: int):
        self.vec_env.batch_reset()

        self.fmc = FMC(self.vec_env) #, policy_model=self.policy_model)
        self.fmc.simulate(max_steps)

    def _get_best_only_batch(self):
        observations = []
        actions = []

        path = self.fmc.tree.best_path
        for state, action in path:
            obs = torch.tensor(state.observation)

            observations.append(obs)
            actions.append(action)

        self.last_episode_total_reward = path.total_reward

        x = torch.stack(observations).float()
        t = torch.tensor(actions)
        return [x], [t]

    def _get_batch(self):
        if not self.fmc.tree:
            raise ValueError("FMC is not tracking walker paths.")

        # TODO: config
        best_only = True

        if best_only:
            return self._get_best_only_batch()

        raise NotImplementedError

        # convert the game tree into a weighted set of targets, based on the walker child counts
        g = self.fmc.tree.g
        for node in g.nodes:
            g.out_edges()

    def train_on_latest_episode(self):
        self.policy_model.train()
        self.optimizer.zero_grad()

        observations, actions = self._get_batch()
        assert len(observations) == len(actions)

        loss = 0
        for obs, action_targets in zip(observations, actions):
            y = self.policy_model.forward(obs, argmax=False)

            # all time steps equal in loss (maximizing average reward)
            trajectory_loss = F.cross_entropy(y, action_targets)
            loss += trajectory_loss

        # average over all trajectories included
        loss /= len(observations)

        loss.backward()
        self.optimizer.step()

        self._log_last_train_step(loss.item())
        return loss.item()

    def evaluate_policy(self, max_steps: int):
        self.policy_model.eval()

        obs = self.env.reset()

        rewards = []

        for _ in range(max_steps):
            action = self.policy_model.forward(obs)
            action = self.policy_model.parse_actions(action)
            obs, reward, done, info = self.env.step(action)
            rewards.append(reward)

        self._log_last_eval_step(rewards)

    def _log_last_train_step(self, train_loss: float):
        if wandb.run is None:
            return

        wandb.log({
            "train/loss": train_loss,
            "train/epsiode_reward": self.last_episode_total_reward,
        })

    def _log_last_eval_step(self, rewards):
        if wandb.run is None:
            return

        wandb.log({
            "eval/total_rewards": sum(rewards),
        })
        
