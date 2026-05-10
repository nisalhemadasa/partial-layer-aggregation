"""
Description: This file consists a simple MLP for image classification.

Author: Nisal Hemadasa
Date: 02-04-2026
Version: 1.0
"""
import torch.nn as nn

import constants


class SimpleModel(nn.Module):
    def __init__(self, _model_id: int = None):
        self.model_id = _model_id

        super(SimpleModel, self).__init__()
        # Define the layers for MNIST dataset (28 x 28 dimensional images)
        self.dense1 = nn.Linear(in_features=28 * 28, out_features=10)  # Dense layer with 28 * 28 inputs and 10 outputs
        self.relu = nn.ReLU()  # ReLU activation function
        self.dense2 = nn.Linear(10, 1)  # Output layer with 10 inputs and 1 output
        # self.sigmoid = nn.Sigmoid()  # Sigmoid activation function

    def forward(self, x):
        """Forward pass through the network"""
        # Flatten the input to (batch_size, 784)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        # x = self.sigmoid(x)
        return x

    def get_model_type(self) -> str:
        """Return the model type as a string."""
        return constants.ModelTypes.SIMPLE_MODEL
