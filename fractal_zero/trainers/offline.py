from copy import copy
import torch
from fractal_zero.data.tree_sampler import TreeSampler

from fractal_zero.search.fmc import FMC
from fractal_zero.loss.space_loss import get_space_loss



class OfflineFMCTrainer:
    def __init__(
        self,
        fmc: FMC,
        policy: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_spec=None,
    ):
        self.fmc = fmc
        self.policy = policy
        self.optimizer = optimizer

        self.space_loss = get_space_loss(self.fmc.vec_env._action_space, loss_spec)

    def simulate_episodes(self, max_steps: int, use_tqdm: bool=False):
        self.fmc.reset()
        self.fmc.simulate(max_steps, use_tqdm=use_tqdm)
        self.tree_sampler = TreeSampler(self.fmc.tree)

    def train_step(self):
        observations, actions, weights = self.tree_sampler.get_batch()
        
        print(len(observations), len(actions), len(weights))
