import gym
import torch
import torch.nn.functional as F
import numpy as np

from typing import Callable, Union
from fractal_zero.data.expert_dataset import ExpertDataset
from fractal_zero.models.joint_model import JointModel
from fractal_zero.search.fmc import FMC

from fractal_zero.vectorized_environment import VectorizedDynamicsModelEnvironment, VectorizedEnvironment, load_environment


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
        representation_model: torch.nn.Module, 
        dynamics_model: torch.nn.Module,
        discriminator_model: torch.nn.Module,
        num_walkers: int,
        action_vectorizer: Callable,
    ):
        self.representation = representation_model
        self.dynamics = dynamics_model
        self.discriminator = discriminator_model

        self.n = num_walkers

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
            raise ValueError("Must call \"set_all_states\" before stepping.")

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

        self.current_reward = self.discriminator.forward(x)  # NOTE: `x` IS THE PREVIOUS STATE!
        self.dones = torch.zeros(x.shape[0], dtype=bool)

        infos = None
        return self.states, self.current_reward, self.dones, infos
    
    def batched_action_space_sample(self):
        action_list = super().batched_action_space_sample()

        # TODO: more general vectorization of actions
        return torch.tensor(action_list, dtype=float).unsqueeze(-1)

    def clone(self, partners, clone_mask):
        self.states[clone_mask] = self.states[partners[clone_mask]]

    def discriminate_single_trajectory(self, observations, embedded_actions):
        # TODO: docstring
        # IMPORTANT NOTE: this forward function does not modify the internal self.states variable of the walkers!

        # can use these for self-consistency loss too :D
        observation_representations = self.representation.forward(observations)
        assert len(observation_representations) == len(embedded_actions)

        steps = embedded_actions.shape[0]
        confusions = torch.zeros(steps, dtype=float)
        self_consistencies = torch.zeros(steps, dtype=float)
        latent_state = observation_representations[0]

        for step in range(steps):
            embedded_action = embedded_actions[step]

            x = torch.cat((latent_state, embedded_action.unsqueeze(0)), dim=-1)

            latent_state = self.dynamics.forward(x)
            
            confusion = self.discriminator.forward(x)
            confusions[step] = confusion

            # self consistency is how well the latent representations match with the representation function
            consistency = F.mse_loss(latent_state, observation_representations[step])
            self_consistencies[step] = consistency

        return confusions, self_consistencies.mean()



class FractalMuZeroDiscriminatorTrainer:

    def __init__(
        self, 
        env: Union[str, gym.Env],
        model_environment: FMZGModel,
        expert_dataset: ExpertDataset,
        optimizer: torch.optim.Optimizer,  # TODO: add check to see if all parameters are inside optimizer (sanity check)
    ):
        # TODO: vectorize the actual environment?
        self.actual_environment = load_environment(env)
        self.model_environment = model_environment

        self.optimizer = optimizer

        # TODO: refac somehow...?
        self.model_environment.action_space = self.actual_environment.action_space

        self.expert_dataset = expert_dataset

    @property
    def discriminator(self):
        return self.model_environment.discriminator

    @property
    def representation(self):
        return self.model_environment.representation

    def _get_agent_trajectory(self, max_steps: int):
        self.model_environment.eval()

        obs = self.actual_environment.reset()
        self.model_environment.set_all_states(obs)

        # TODO: maybe incorporate policy model? or maybe we can just use FMC to search?
        self.fmc = FMC(self.model_environment)

        lookahead_steps = 16

        observations = []
        actions = []

        for _ in range(max_steps):
            self.fmc.reset()

            observations.append(torch.tensor(obs, dtype=float))

            action = self.fmc.simulate(lookahead_steps)
            action = self.model_environment.action_vectorizer(action)

            actions.append(action)

            obs, reward, done, info = self.actual_environment.step(action)
            self.model_environment.set_all_states(obs)

            if done:
                break

        x = torch.stack(observations)
        y = torch.tensor(actions)

        return x, y

    def _get_expert_batch(self):
        # TODO
        raise NotImplementedError

    def _discriminator_train_step(self):
        # TODO
        raise NotImplementedError

    def train_step(self, max_steps: int):
        self.model_environment.train()
        self.optimizer.zero_grad()

        # TODO: simplify this, lots of copies!
        # get batch
        agent_x, agent_y = self._get_agent_trajectory(max_steps)
        expert_x, expert_y = self.expert_dataset.sample_trajectory(max_steps)

        # # get hidden representation of the observations as states
        # agent_states = self.representation.forward(agent_x.float())
        # expert_states = self.representation.forward(expert_x.float())

        # # add the hidden representations with the action embeddings (TODO: de-duplciate this code, it exists
        # # within the FMZG model too.)
        # agent_samples = torch.cat((agent_states, agent_y.unsqueeze(-1)), dim=-1)
        # expert_samples = torch.cat((expert_states, expert_y.unsqueeze(-1)), dim=-1)
        # x = torch.cat((agent_samples, expert_samples)).float()
        # t = torch.tensor(([0] * agent_samples.shape[0]) + [1] * expert_samples.shape[0]).float()

        # discriminator_confusions = self.discriminator.forward(x).squeeze(-1)

        # TODO: write discriminator forward such that it can support full trajectories (if using a transformer/RNN)

        assert len(agent_x) == len(agent_y)
        assert len(expert_x) == len(expert_y)

        agent_confusions, agent_consistency = self.model_environment.discriminate_single_trajectory(
            agent_x.float(), 
            agent_y.float(),
        )
        expert_confusions, expert_consistency = self.model_environment.discriminate_single_trajectory(
            expert_x.float(), 
            expert_y.float(),
        )

        assert len(agent_confusions) == len(agent_x)

        agent_t = torch.zeros(agent_x.shape[0], dtype=float)
        expert_t = torch.ones(expert_x.shape[0], dtype=float)

        loss = 0

        loss += F.mse_loss(agent_confusions, agent_t)
        loss += F.mse_loss(expert_confusions, expert_t)

        # TODO: config for self consistency loss
        # loss += (agent_consistency + expert_consistency) / 2
        
        loss.backward()
        self.optimizer.step()

        return loss.item()