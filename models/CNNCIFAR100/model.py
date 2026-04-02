"""
Description: This file consists a simple CNN for image classification, for CIFAR 10/CIFAR 100 dataset (32 x 32
dimensional images) in 3 channels

Author: Nisal Hemadasa
Date: 02-04-2026
Version: 1.0
"""

import torch
import torch.nn as nn

import constants


class CNNCIFAR100(nn.Module):
    """
    A Convolutional Neural Network for CIFAR-100 dataset.
    Input: 32x32 RGB images (3 channels)
    Output: 100-class(CIFAR-100) classification
    """

    def __init__(self, num_classes=100, _model_id: int = None):
        self.model_id = _model_id

        super(CNNCIFAR100, self).__init__()

        # CIFAR has 3 channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        # CIFAR feature-map output size calculation:
        # Input: 32x32
        # After conv1 (no padding):   30x30
        # After maxpool:              15x15
        # After conv2:                13x13
        # After maxpool:               6x6
        # Feature size = 64 * 6 * 6

        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return torch.log_softmax(x, dim=1)

    def get_model_type(self) -> str:
        """Return the model type as a string."""
        return constants.ModelTypes.CNN_CIFAR_100
