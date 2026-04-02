"""
Description: This file consists a simple CNN for image classification, for TinyImageNet-200 dataset.
Input:  [B, 3, 64, 64]

Author: Nisal Hemadasa
Date: 02-04-2026
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import constants


class ResidualBlock(nn.Module):
    """Small ResNet-like block used in a lightweight Tiny ImageNet CNN."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out, inplace=True)
        return out


class CNNTinyImageNet(nn.Module):
    """
    Lightweight ResNet-style CNN for Tiny ImageNet-200.

    Input:  [B, 3, 64, 64]
    Output: [B, num_classes] (logits, not log_softmax)
    """

    def __init__(self, num_classes: int = 200, _model_id: int = None):
        self.model_id = _model_id

        super().__init__()

        # Stem: keep it small
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),  # [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),  # [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),  # [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),  # [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),  # [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 32 channels, spatial 64x64 -> 32x32
        self.block1 = ResidualBlock(32, 32, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 -> 32x32

        # Stage 2: 32 -> 64 channels, 32x32 -> 16x16
        self.block2 = ResidualBlock(32, 64, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16

        # self.dropout = nn.Dropout(p=0.5)
        # self.fc = nn.Linear(64, num_classes)

        # Stage 3: 64 -> 128 channels, 16x16 -> 8x8
        self.block3 = ResidualBlock(64, 128, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8

        # Global average pooling instead of a huge FC
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, 3, 64, 64]
        x = self.stem(x)  # [B, 32, 64, 64]

        x = self.block1(x)  # [B, 32, 64, 64]
        x = self.pool1(x)  # [B, 32, 32, 32]

        x = self.block2(x)  # [B, 64, 32, 32]
        x = self.pool2(x)  # [B, 64, 16, 16]

        x = self.block3(x)  # [B, 128, 16, 16]
        x = self.pool3(x)  # [B, 128, 8, 8]

        # Global average pooling to [B, 128]
        x = F.adaptive_avg_pool2d(x, output_size=1)  # [B, 128, 1, 1]
        x = torch.flatten(x, 1)  # [B, 128]

        x = self.dropout(x)
        x = self.fc(x)  # [B, num_classes]

        # For CrossEntropyLoss, logits are preferred (no log_softmax here)
        return x

    def get_model_type(self) -> str:
        """Return the model type as a string."""
        return constants.ModelTypes.CNN_TINY_IMAGENET

