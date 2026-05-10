"""
Description: This file consists a simple CNN for image classification, for CIFAR 10/CIFAR 100 dataset (32 x 32
dimensional images) in 3 channels.

Author: Nisal Hemadasa
Date: 02-04-2026
Version: 1.0
"""

import torch
import torch.nn as nn

import constants


class CNNCIFAR10(nn.Module):
    """
    A Convolutional Neural Network for CIFAR-10 dataset.
    Input: 32x32 RGB images (3 channels)
    Output: 10-class(CIFAR-10) classification
    """

    def __init__(self, num_classes=10, _model_id: int = None):
        self.model_id = _model_id

        super(CNNCIFAR10, self).__init__()

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
        # self.fc2 = nn.Linear(in_features=128, out_features=64)  # this line is added to experiment layer removal case 4 of FedEx
        # self.fc3 = nn.Linear(in_features=64, out_features=num_classes)   # this line is added to experiment layer removal case 4 of FedEx
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
        # x = torch.relu(self.fc2(x))    # this line is added to experiment layer removal case 4 of FedEx
        # x = self.fc3(x)    # this line is added to experiment layer removal case 4 of FedEx

        return torch.log_softmax(x, dim=1)

    def get_model_type(self) -> str:
        """Return the model type as a string."""
        return constants.ModelTypes.CNN_CIFAR_10


class CNNCIFAR10Deep(nn.Module):
    """
    Deeper CNN for CIFAR-10.
    Input: 32x32 RGB images
    Output: 10-class classification
    """

    def __init__(self, num_classes=10, _model_id: int = None):
        super(CNNCIFAR10Deep, self).__init__()
        self.model_id = _model_id

        # Block 1: 32x32 -> 16x16
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 2: 16x16 -> 8x8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        # Block 3: 8x8 -> 4x4
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)

        # Feature size after three 2x2 pools:
        # 32 -> 16 -> 8 -> 4
        # 256 * 4 * 4 = 4096
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 128)
        self.bn_fc3 = nn.BatchNorm1d(128)

        self.fc4 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Block 1
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        # Block 2
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        # Block 3
        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.relu(self.bn6(self.conv6(x)))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)

        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)

        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)

        x = torch.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout(x)

        x = self.fc4(x)

        return torch.log_softmax(x, dim=1)

    def get_model_type(self) -> str:
        return constants.ModelTypes.CNN_CIFAR_10
