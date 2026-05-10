"""
Description: This file consists a simple model for binary classification of ADULT data (tabular)

Author: Nisal Hemadasa
Date: 28-04-2026
Version: 1.0
"""

import torch
import torch.nn as nn

import constants


class AdultClassifier(nn.Module):
    """
    Feed-forward neural network for binary classification on the Adult dataset.

    Architecture follows the model used for Adult-GDrift in the referenced paper:
    a fully connected neural network with one hidden layer containing 10 tanh
    units, followed by a single sigmoid output neuron for binary classification.

    Input:
        x: tabular Adult features, shape [batch_size, input_dim]

    Output:
        Probability of the positive class, shape [batch_size, 1]

    Naming convention:
        - `fc1.*` belongs to the feature extractor / hidden representation.
        - `fc2.*` is the classifier head, so it can be selected using:
          k.startswith("fc2.")
    """

    def __init__(self, input_dim: int, _model_id: int = None):
        super().__init__()

        self.model_id = _model_id

        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def get_model_type(self) -> str:
        """Return the model type as a string."""
        return constants.ModelTypes.ADULT_CLASSIFIER