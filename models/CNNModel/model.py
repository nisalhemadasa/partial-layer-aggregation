"""
Description: This file consists a CNN model for image classification, for MNIST and F_MNIST dataset (28 x 28
dimensional images)

Author: Nisal Hemadasa
Date: 02-04-2026
Version: 1.0
"""

import torch
import torch.nn as nn

import constants


class CNNModel(nn.Module):
    def __init__(self, _model_id: int = None):
        self.model_id = _model_id
        super(CNNModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=32 * 13 * 13, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        # # Fully connected layers
        # self.fc1 = nn.Linear(in_features=64 * 5 * 5, out_features=128)
        # self.fc2 = nn.Linear(in_features=128, out_features=10)
        # self.fc2 = nn.Linear(in_features=128, out_features=64)  # this line is added to experiment layer removal case 4 of FedEx
        # self.fc3 = nn.Linear(in_features=64, out_features=10)   # this line is added to experiment layer removal case 4 of FedEx

        # Regularization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """Describes how input data flows through the network"""
        # Convolutional feature extractor
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        # Flatten
        x = x.view(-1, 32 * 13 * 13)

        # x = torch.relu(self.conv2(x))
        # x = torch.max_pool2d(x, kernel_size=2, stride=2)
        # # Flatten
        # x = x.view(-1, 64 * 5 * 5)

        # Fully connected classifier
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # x = torch.relu(self.fc2(x))    # this line is added to experiment layer removal case 4 of FedEx
        # x = self.fc3(x)    # this line is added to experiment layer removal case 4 of FedEx

        return torch.log_softmax(x, dim=1)

    def get_model_type(self) -> str:
        """Return the model type as a string."""
        return constants.ModelTypes.CNN_MODEL

