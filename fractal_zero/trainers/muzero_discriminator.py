import gym
from matplotlib.pyplot import xlim
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from typing import Callable, Union
from fractal_zero.data.expert_dataset import ExpertDataset
from fractal_zero.loss.space_loss import get_space_loss
from fractal_zero.models.joint_model import JointModel
from fractal_zero.search.fmc import FMC

import wandb

from fractal_zero.vectorized_environment import (
    VectorizedDynamicsModelEnvironment,
    VectorizedEnvironment,
    load_environment,
)


class FMZGModel(VectorizedEnvironment):
    action_space: gym.Space

    # TODO: actual docstring:
    # the representation model takes the raw obesrvation and puts it into an embedding. this representation model MAY be
    #   a transformer or some sort of recurrent model, in the future.
    # the dynamics model is given the state and action embeddings and returns a new state embedding
    # the discriminator model is given the embedding of the new state (from the dynamics model) and returns
    #   a float reward between 0 and 1 (0=agent, 1=expert)

    def __init__(
        self,
        env: Union[str, gym.Env],
        representation_model: torch.nn.Module,
        dynamics_model: torch.nn.Module,
        discriminator_model: torch.nn.Module,
        num_walkers: int,
        action_vectorizer: Callable,
    ):
        super().__init__(env, num_walkers)

        self.representation = representation_model
        self.dynamics = dynamics_model
        self.discriminator = discriminator_model

        # TODO: refac?
        self.action_vectorizer = action_vectorizer

        self.initial_states = None
        self.states = None
        self.current_reward = None
        self.dones = None

    def eval(self):
        self.representation.eval()
        self.discriminator.eval()
        self.dynamics.eval()

    def train(self):
        self.representation.train()
        self.discriminator.train()
        self.dynamics.train()

    def _check_states(self):
        if self.initial_states is None:
            raise ValueError('Must call "set_all_states" before stepping.')

    def batch_reset(self):
        # NOTE: does nothing...?

        self._check_states()
        return self.states

    def set_all_states(self, observation):
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=float)

        self.initial_states = self.representation.forward(observation.float())

        # duplicate initial state representation to all walkers
        self.states = torch.zeros((self.n, *self.initial_states.shape))

        self.states[:] = self.initial_states

    def batch_step(self, embedded_actions):
        self._check_states()

        # update to new state
        x = torch.cat((self.states.float(), embedded_actions.float()), dim=-1)
        self.states = self.dynamics.forward(x)

        self.current_reward = self.discriminator.forward(
            x
        )  # NOTE: `x` IS THE PREVIOUS STATE!
        self.dones = torch.zeros(x.shape[0], dtype=bool)

        infos = None
        observations = self.states
        return self.states, observations, self.current_reward, self.dones, infos

    def batched_action_space_sample(self):
        action_list = super().batched_action_space_sample()

        # TODO: more general vectorization of actions
        return torch.tensor(action_list, dtype=float).unsqueeze(-1)

    def clone(self, partners, clone_mask):
        self.states[clone_mask] = self.states[partners[clone_mask]]

    def get_state_embeddings_for_trajectory(self, observations, embedded_actions):
        # TODO: docstring
        # IMPORTANT NOTE: this forward function does not modify the internal self.states variable of the walkers!

        # can use these for self-consistency loss too :D
        observation_representations = self.representation.forward(observations)
        assert len(observation_representations) == len(embedded_actions)

        steps = embedded_actions.shape[0]
        # confusions = torch.zeros(steps, dtype=float)
        self_consistencies = torch.zeros(steps, dtype=float)
        latent_state = observation_representations[0]

        step_embeddings = torch.zeros((steps, latent_state.numel() + embedded_actions[0].numel()), dtype=float).float()

        for step in range(steps):
            embedded_action = embedded_actions[step]

            x = torch.cat((latent_state, embedded_action.unsqueeze(0)), dim=-1)
            step_embeddings[step] = x  # store for head calculations

            latent_state = self.dynamics.forward(x)
            

            # confusion = self.discriminator.forward(x)
            # confusions[step] = confusion

            # self consistency is how well the latent representations match with the representation function
            consistency = F.mse_loss(latent_state, observation_representations[step])
            self_consistencies[step] = consistency

        return step_embeddings, self_consistencies


class FractalMuZeroDiscriminatorTrainer:
    def __init__(
        self,
        env: Union[str, gym.Env],
        model_environment: FMZGModel,
        policy_model: torch.nn.Module,
        expert_dataset: ExpertDataset,
        optimizer: torch.optim.Optimizer,  # TODO: add check to see if all parameters are inside optimizer (sanity check)
        loss_spec=None,
    ):
        # TODO: vectorize the actual environment?
        self.actual_environment = load_environment(env)
        self.model_environment = model_environment
        self.policy_head = policy_model

        self.optimizer = optimizer

        # TODO: refac somehow...?
        self.action_space = self.actual_environment.action_space
        self.model_environment.action_space = self.action_space
        self.action_space_loss = get_space_loss(self.action_space, loss_spec)

        self.expert_dataset = expert_dataset

        self.best_policy = None
        self.most_reward = float("-inf")

    @property
    def discriminator_head(self):
        return self.model_environment.discriminator

    @property
    def representation(self):
        return self.model_environment.representation

    def _get_agent_trajectory(self, max_steps: int, render: bool = False):
        self.model_environment.eval()

        obs = self.actual_environment.reset()
        self.model_environment.set_all_states(obs)

        # TODO: maybe incorporate policy model? or maybe we can just use FMC to search?

        lookahead_steps = 64

        observations = []
        actions = []

        for _ in range(max_steps):
            self.fmc = FMC(self.model_environment)

            observations.append(torch.tensor(obs, dtype=float))

            action = self.fmc.simulate(lookahead_steps)
            action = self.model_environment.action_vectorizer(action)

            actions.append(action)

            obs, reward, done, info = self.actual_environment.step(action)
            self.model_environment.set_all_states(obs)

            if render:
                self.actual_environment.render()

            if done:
                break

        x = torch.stack(observations)
        y = torch.tensor(actions)

        return x, y

    def _get_agent_batch(self, batch_size: int, max_steps: int):
        observations = []
        actions = []
        labels = []
        for _ in range(batch_size):
            agent_x, agent_y = self._get_agent_trajectory(max_steps)
            observations.append(agent_x)
            actions.append(agent_y)
            labels.append(torch.zeros(agent_x.shape[0], dtype=float))
        return observations, actions, labels

    def generate_batches(self, max_steps: int):
        self.model_environment.eval()
        batch_size_per_class = 16   # TODO: config
        self.agent_batch = self._get_agent_batch(batch_size_per_class, max_steps)
        self.expert_batch = self.expert_dataset.sample_batch(batch_size_per_class, max_steps)

        if wandb.run:
            amean_steps = np.mean([len(o) for o in self.agent_batch[0]])
            emean_steps = np.mean([len(o) for o in self.expert_batch[0]])
            wandb.log({
                "batches/agent_mean_steps": amean_steps,
                "batches/expert_mean_steps": emean_steps,
            })

    # def _get_discriminator_loss(self, batch):
    #     self.model_environment.train()

    #     c = 0
    #     loss = 0
    #     for x, y, labels in zip(*batch):
    #         # TODO: config for self consistency loss
    #         (
    #             confusions,
    #             consistency,
    #         ) = self.model_environment.discriminate_single_trajectory(x.float(), y.float())
    #         loss += F.mse_loss(confusions, labels)
    #         c += 1
    #     return loss / c

    # def _get_policy_loss(self, batch):
    #     self.policy_model.train()

    #     c = 0
    #     loss = 0
    #     for x, t, _ in zip(*batch):
    #         y = self.policy_model(x.float())
    #         loss += self.action_space_loss(y, t)
    #         c += 1
    #     return loss / c

    def _get_losses(self, batch):
        self.model_environment.train()
        self.policy_head.train()

        c = 0
        discriminator_loss = 0
        policy_loss = 0
        for observations, embedded_actions, discriminator_labels in zip(*batch):
            # TODO: use consistencies somehow?

            state_embeddings, consistencies = self.model_environment.get_state_embeddings_for_trajectory(
                observations.float(), 
                embedded_actions.float(),
            )

            policy_predictions = self.policy_head(state_embeddings)
            discrim_confusions = self.discriminator_head(state_embeddings)

            policy_loss += self.action_space_loss(policy_predictions, embedded_actions.float())
            discriminator_loss += F.mse_loss(discrim_confusions, discriminator_labels.float())
            c += 1

        discriminator_loss /= c
        policy_loss /= c

        return discriminator_loss, policy_loss


    def train_step(self):
        self.model_environment.train()
        self.optimizer.zero_grad()

        # TODO: instead of having the discriminator discriminate directly against FMC,
        # have it discriminate against a
        #  policy model being trained by FMC?
        # then, the discriminator should train to predict the differences between the policy model
        # and the expert model.
        # TODO: both should OPTIONALLY share the dynamics function backbone. if the gen and discrim should NOT have
        # a shared backbone, the policy model's dynamics function should be used by FMC.
        
        # loss = 0

        # agent_loss = self._get_discriminator_loss(self.agent_batch)
        # expert_loss = self._get_discriminator_loss(self.expert_batch)
        # discriminator_loss = (agent_loss + expert_loss) / 2
        # loss += discriminator_loss

        # agent_policy_loss = self._get_policy_loss(self.agent_batch)
        # expert_policy_loss = self._get_policy_loss(self.expert_batch)
        # policy_loss = (agent_policy_loss + expert_policy_loss) / 2
        # loss += policy_loss
        # loss += agent_policy_loss

        # TODO: explain
        agent_discrim_loss, agent_policy_loss = self._get_losses(self.agent_batch)
        expert_discrim_loss, expert_policy_loss = self._get_losses(self.expert_batch)

        discriminator_loss = (agent_discrim_loss + expert_discrim_loss) / 2
        policy_loss = agent_policy_loss  # TODO: use expert actions for loss too?

        loss = discriminator_loss + policy_loss

        loss.backward()
        self.optimizer.step()

        if wandb.run:
            wandb.log({
                "discriminator/train_loss": discriminator_loss,
                "discriminator/agent_loss": agent_discrim_loss,
                "discriminator/expert_loss": expert_discrim_loss,
                "policy/agent_loss": agent_policy_loss,
                "policy/expert_loss": expert_policy_loss,
            })

        return discriminator_loss.item()

    # @torch.no_grad()
    # def evaluate_step(self, max_steps: int):
    #     self.model_environment.eval()
    #     self.policy_head.eval()

    #     rewards = []

    #     obs = self.actual_environment.reset()
    #     for step in range(max_steps):
    #         # TODO: this needs to be refactored for sure
    #         x = torch.tensor(obs).float()
    #         x = self.model_environment.representation(x)
    #         action = self.policy_head(x)
    #         action = self.policy_head.parse_action(action)

    #         obs, reward, done, info = self.actual_environment.step(action)
    #         rewards.append(reward)
    #         if done:
    #             break

    #     total_reward = sum(rewards)
    #     if total_reward > self.most_reward:
    #         self.most_reward = total_reward
    #         self.best_policy = deepcopy(self.policy_head)

    #     if wandb.run:
    #         wandb.log({
    #             "eval/total_rewards": total_reward,
    #             "eval/episode_length": step,
    #         })