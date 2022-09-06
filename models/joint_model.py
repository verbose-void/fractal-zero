import torch

from models.dynamics import FullyConnectedDynamicsModel
from models.prediction import FullyConnectedPredictionModel
from models.representation import FullyConnectedRepresentationModel

class JointModel(torch.nn.Module):
    def __init__(self, representation_model: FullyConnectedRepresentationModel, dynamics_model: FullyConnectedDynamicsModel, prediction_model: FullyConnectedPredictionModel):
        super().__init__()

        # TODO: base classes for rep/dyn/pred models
        self.representation_model = representation_model
        self.dynamics_model = dynamics_model
        self.prediction_model = prediction_model