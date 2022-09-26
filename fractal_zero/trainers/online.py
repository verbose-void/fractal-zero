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

    def _get_best_history(self) -> GameHistory:
        return max(self.fmc.game_histories, key=lambda h: h.total_reward)
        
    def _get_batch(self):
        history = self._get_best_history()

        self.last_episode_total_reward = history.total_reward

        x = torch.stack(history.observations[1:]).float()
        t = torch.tensor(history.actions[1:])
        return x, t.long()

    def generate_episode_data(self, max_steps: int):
        self.vec_env.batch_reset()

        self.fmc = FMC(self.vec_env, policy_model=self.policy_model)
        self.fmc.simulate(max_steps)

    def train_on_latest_episode(self):
        self.policy_model.train()
        self.optimizer.zero_grad()

        x, t = self._get_batch()
        y = self.policy_model.forward(x, argmax=False)

        loss = F.cross_entropy(y, t)
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
        
