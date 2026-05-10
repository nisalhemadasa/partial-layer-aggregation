"""
Description: This file consists a simple CNN for image classification, for CIFAR 10/CIFAR 100 dataset (32 x 32
dimensional images) in 3 channels

Author: Nisal Hemadasa
Date: 02-04-2026
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes * self.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18CIFAR100(nn.Module):
    """
    ResNet-18-style model adapted for CIFAR-100.

    Input:  32x32 RGB images
    Output: 100-class CIFAR-100 classification

    Important:
    - Feature extractor parameters are everything except `fc2.*`
    - Classifier parameters are named `fc2.*`
    """

    def __init__(self, num_classes=100, _model_id: int = None):
        super().__init__()

        self.model_id = _model_id
        self.in_planes = 64

        # CIFAR-style stem: 3x3 conv, stride 1, no maxpool
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet-18 stages: [2, 2, 2, 2]
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Keep this name for your existing classifier split logic
        self.fc2 = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def get_model_type(self) -> str:
        """Return the model type as a string."""
        return constants.ModelTypes.CNN_CIFAR_100


class ShallowResNetCIFAR100(nn.Module):
    """
    Shallow ResNet-style model for CIFAR-100.

    This model keeps the residual-learning structure of ResNet but uses a much
    smaller feature extractor than ResNet-18. Instead of four residual stages
    with [2, 2, 2, 2] BasicBlocks, it uses only two BasicBlocks, giving four
    convolutional layers inside the residual extractor. This reduces the
    dominance of the backbone relative to the one-layer classifier head `fc2`,
    which is useful in PaLA/FedEx-style settings where the classifier is kept
    local or treated separately from the aggregated extractor.

    Input:  32x32 RGB images
    Output: 100-class CIFAR-100 classification

    Classifier head:
        `fc2.*`
    """

    def __init__(self, num_classes=100, _model_id: int = None):
        super().__init__()

        self.model_id = _model_id
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        # Much shorter extractor: 2 BasicBlocks = 4 conv layers
        self.layer1 = self._make_layer(BasicBlock, 64, 1, stride=1)
        # self.layer2 = self._make_layer(BasicBlock, 128, 1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Keep classifier name for your split logic
        self.fc2 = nn.Linear(64 * BasicBlock.expansion, num_classes)
        # self.fc2 = nn.Linear(128 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        # x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def get_model_type(self) -> str:
        """Return the model type as a string."""
        return constants.ModelTypes.CNN_CIFAR_100
