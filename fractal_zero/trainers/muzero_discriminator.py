import gym
import torch
import numpy as np

from typing import Union
from fractal_zero.models.joint_model import JointModel

from fractal_zero.vectorized_environment import VectorizedDynamicsModelEnvironment, VectorizedEnvironment, load_environment


class FMZGModel(VectorizedEnvironment):

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
    ):
        self.representation = representation_model
        self.dynamics = dynamics_model
        self.discriminator = discriminator_model

        self.n = num_walkers

        self.initial_states = None
        self.states = None
        self.current_reward = None

    def set_all_states(self, observation: np.ndarray):
        self.initial_states = self.representation.forward(observation)

        # duplicate initial state representation to all walkers
        self.states = torch.zeros((self.n, *self.initial_states.shape))
        self.states[:] = self.initial_states

    def batch_step(self, embedded_actions):
        if self.initial_states is None:
            raise ValueError("Must call \"set_all_states\" before stepping.")

        # update to new state
        self.states = self.dynamics.forward(self.states, embedded_actions)

        self.current_reward = self.discriminator.forward(self.states)
        return self.states, self.current_reward


class FractalMuZeroDiscriminatorTrainer:

    def __init__(
        self, 
        env: Union[str, gym.Env],
        model: FMZGModel,
    ):
        # TODO: vectorize the actual environment?
        self.actual_environment = load_environment(env)
        self.model = model

        # TODO: maybe incorporate policy model? or maybe we can just use FMC to search?
        self.fmc = None  # TODO

        self.expert_dataset = None  # TODO

    def _get_agent_batch(self):
        # TODO
        raise NotImplementedError

    def _get_expert_batch(self):
        # TODO
        raise NotImplementedError

    def discriminator_train_step(self):
        # TODO
        raise NotImplementedError
