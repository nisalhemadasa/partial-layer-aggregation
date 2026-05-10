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


class ResNet18TinyImageNet(nn.Module):
    """A ResNet-18 style model adapted for Tiny ImageNet (64x64 inputs)."""

    def __init__(self, num_classes: int = 200, _model_id: int = None):
        super().__init__()
        self.model_id = _model_id

        # Small-image-friendly ResNet-18 stem: use 3x3 conv, stride=1 and no maxpool
        # (ImageNet-style 7x7+pool is too aggressive for 64x64 inputs)
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # no maxpool here

        # Layers: each layer contains 2 ResidualBlocks (ResNet-18)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, num_classes)

        # Weight initialization: Kaiming for convs, ones for bn weights, zeros for biases
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        # First block may downsample
        layers.append(ResidualBlock(self.in_channels, out_channels, stride=stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, 3, 64, 64]
        x = self.conv1(x)   # -> [B, 64, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        # no maxpool here for small images; keep higher spatial resolution
        # after conv1 + bn1 + relu: [B, 64, 64, 64]

        x = self.layer1(x)  # -> [B, 64, 16, 16]
        x = self.layer2(x)  # -> [B, 128, 8, 8]
        x = self.layer3(x)  # -> [B, 256, 4, 4]
        x = self.layer4(x)  # -> [B, 512, 2, 2]

        x = self.avgpool(x)  # -> [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # -> [B, 512]
        x = self.dropout(x)
        x = self.fc(x)  # -> [B, num_classes]
        return x

    def get_model_type(self) -> str:
        return "ResNet18TinyImageNet"


class ConvNeXtBlock(nn.Module):
    """A minimal ConvNeXt block implementation adapted for small images."""

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        # depthwise conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # pointwise MLP implemented as linear layers applied per-channel after permuting to channels-last
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma = None

        self.drop_path = nn.Identity()  # no stochastic depth for now

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        # to channels-last for LayerNorm
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXtTinyTinyImageNet(nn.Module):
    """ConvNeXt-Tiny-like model adapted for TinyImageNet (64x64).

    This is a reduced ConvNeXt design with smaller channel widths and depths to fit compute.
    """

    def __init__(self, num_classes: int = 200, _model_id: int = None):
        super().__init__()
        self.model_id = _model_id

        # Use smaller channel sizes appropriate for resource constraints
        dims = [32, 64, 128, 256]
        depths = [2, 2, 6, 2]

        # stem: 4x4 conv stride 4 to downsample 64->16
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )

        # stages and downsamples
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(len(depths)):
            blocks = [ConvNeXtBlock(dims[i]) for _ in range(depths[i])]
            self.stages.append(nn.Sequential(*blocks))
            if i < len(depths) - 1:
                self.downsamples.append(nn.Sequential(
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2, bias=False),
                    nn.BatchNorm2d(dims[i+1]),
                ))

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 64, 64]
        # stem: 4x4 conv stride 4 -> [B, 32, 16, 16]
        x = self.stem(x)

        # stages: run each stage then optionally downsample
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        x = self.global_pool(x)  # [B, C, 1, 1]
        x = torch.flatten(x, 1)  # [B, C]
        # LayerNorm expects last dimension features
        x = self.norm(x)
        x = self.head(x)
        return x

    def get_model_type(self) -> str:
        return "ConvNeXtTinyTinyImageNet"


