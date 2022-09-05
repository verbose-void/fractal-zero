import torch

class JointModel(torch.nn.Module):
    def __init__(self, representation_model, dynamics_model):
        super().__init__()

        self.representation_model = representation_model
        self.dynamics_model = dynamics_model