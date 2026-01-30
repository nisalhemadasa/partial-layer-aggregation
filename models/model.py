"""
Description: This module defines feedforward neural network models for image classification using PyTorch.

Author: Nisal Hemadasa
Date: 04-04-2025
Version: 2.0
"""
from abc import ABC, abstractmethod
from typing import Tuple, OrderedDict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import constants

DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__}"
)


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


# for MNIST and F_MNIST dataset (28 x 28 dimensional images)
class CNNModel(nn.Module):
    def __init__(self, _model_id: int = None):
        self.model_id = _model_id

        """Defines the architecture of the network"""
        super(CNNModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

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


# for CIFAR 10/CIFAR 100 dataset (32 x 32 dimensional images) in 3 channels
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


# for CIFAR 10/CIFAR 100 dataset (32 x 32 dimensional images) in 3 channels
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


# class CNNTinyImageNet(nn.Module):
#     """
#     A Convolutional Neural Network for Tiny ImageNet-200.
#
#     Input:  64x64 RGB images (3 channels)
#     Output: 200-class classification (Tiny ImageNet-200)
#     """
#
#     def __init__(self, num_classes: int = 200):
#         super(CNNTinyImageNet, self).__init__()
#
#         # Tiny ImageNet / CIFAR-style backbone
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)  # 64x64 -> 62x62
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)  # 31x31 -> 29x29
#
#         # For 64x64 input:
#         #  conv1: 64x64 -> 62x62
#         #  pool:  62x62 -> 31x31
#         #  conv2: 31x31 -> 29x29
#         #  pool:  29x29 -> 14x14
#         # So flattened feature size = 64 * 14 * 14 = 12544
#         self.fc1 = nn.Linear(64 * 14 * 14, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#
#         self.dropout = nn.Dropout(p=0.5)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [batch, 3, 64, 64]
#         x = F.relu(self.conv1(x))  # -> [B, 32, 62, 62]
#         x = F.max_pool2d(x, kernel_size=2, stride=2)  # -> [B, 32, 31, 31]
#
#         x = F.relu(self.conv2(x))  # -> [B, 64, 29, 29]
#         x = F.max_pool2d(x, kernel_size=2, stride=2)  # -> [B, 64, 14, 14]
#
#         x = x.view(x.size(0), -1)  # -> [B, 64*14*14 = 12544]
#
#         x = F.relu(self.fc1(x))  # -> [B, 128]
#         x = self.dropout(x)
#         x = self.fc2(x)  # -> [B, num_classes]
#
#         return F.log_softmax(x, dim=1)
#
#     def get_model_type(self) -> str:
#         """Return the model type as a string."""
#         return constants.ModelTypes.CNN_TINY_IMAGENET
class ResidualBlock(nn.Module):
    """Small ResNet-like block used in a lightweight Tiny ImageNet CNN."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Optional 1x1 conv to match shape for the skip connection
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
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


def split_to_extractor_and_classifier(_model: nn.Module, _model_params: OrderedDict) -> tuple[OrderedDict, OrderedDict]:
    """
    Split the model/model parameters into feature extractor and classifier parts. The feature extractor includes all layers except the
    final fully connected layer (fc2), while the classifier includes only the final fully connected layer.
    :param _model: The model to split
    :param _model_params: The model parameters to split
    :return: A tuple containing two OrderedDicts: (parameters of the feature extractor, parameters of the classifier)
    """
    if _model is not None:
        _model_params = _model.state_dict()

    # # case 1
    # get the extractor parameters (all except fc2 layer)
    # extractor_params = OrderedDict((k, v) for k, v in _model_params.items() if not k.startswith("fc1."))
    # # get the extractor parameters (all except fc2 layer)
    # extractor_params = OrderedDict((k, v) for k, v in _model_params.items() if not k.startswith("fc2."))
    # # For Tiny ImageNet200 model
    # extractor_params = OrderedDict((k, v) for k, v in _model_params.items() if not k.startswith("block3."))

    # # case 2
    # get the extractor parameters (all except fc2 and fc1 layer)
    extractor_params = OrderedDict((k, v) for k, v in _model_params.items() if
                                   not (k.startswith("fc1.") or k.startswith("fc2.") or k.startswith('conv2')))
    # # For Tiny ImageNet200 model
    # extractor_params = OrderedDict((k, v) for k, v in _model_params.items() if
    #                                not (k.startswith("block3.") or k.startswith("block2.") or k.startswith("block1.")))
    # extractor_params = OrderedDict((k, v) for k, v in _model_params.items() if
    #                                not (k.startswith("block3.") or k.startswith("block2.")))

    # # case 3
    # # get the extractor parameters (only fc2 layer)
    # extractor_params = OrderedDict((k, v) for k, v in _model_params.items() if k.startswith("fc2."))

    # # case 4
    # # No layers are dropped. Layer (fc3.) is added.
    # extractor_params = OrderedDict((k, v) for k, v in _model_params.items())

    # # case 5
    # # get the extractor parameters (all except fc3. layer)
    # extractor_params = OrderedDict((k, v) for k, v in _model_params.items() if not k.startswith("fc3."))

    # # case 6
    # # get the extractor parameters (only fc3. layer)
    # extractor_params = OrderedDict((k, v) for k, v in _model_params.items() if k.startswith("fc3."))

    # # case 7
    # # get the extractor parameters (only fc2. and fc3. layer)
    # extractor_params = OrderedDict((k, v) for k, v in _model_params.items() if (k.startswith("fc3.") or k.startswith("fc2.")))

    # # case 8
    # # get the extractor parameters (all except fc2. layer: middle layer), fc1, fc3 remaining.
    # extractor_params = OrderedDict((k, v) for k, v in _model_params.items() if not k.startswith("fc2."))

    # get the classifier parameters (fc2 layer)
    # classifier_params = OrderedDict((k, v) for k, v in _model_params.items() if k.startswith("fc2."))
    # For Tiny ImageNet200 model
    classifier_params = OrderedDict((k, v) for k, v in _model_params.items() if k.startswith("block3."))

    return extractor_params, classifier_params


def fedau_clientside_train(_model: nn.Module, _dataset: DataLoader, _aux_dataset: DataLoader,
                           _server_model_params: OrderedDict, _drifted_client_indices: List[int], _client_id,
                           _epochs: int, _mini_batch_size: int) -> OrderedDict:
    """
    Performs clients dei training operations of the FedAU algorithm, following [2]. 
    This includes training the (1) learning module and (2) auxiliary module
    
    [2] Implementation of FedAU algorithm following the paper: [2]H. Gu, G. Zhu, J. Zhang, X. Zhao, Y. Han, L. Fan and
    Q. Yang, “Unlearning during Learning: An Efficient Federated Machine Unlearning Method,” in
    Proceedings of the 33rd International Joint Conference on Artificial Intelligence (IJCAI-24)
    :param _model: Core learning model
    :param _dataset: The dataloader containing training dataset
    :param _aux_dataset: The dataloader containing auxiliary training dataset
    :param _server_model_params: The server model parameters (weights and biases from the server)
    :param _client_id: ID of the given client
    :param _drifted_client_indices: List of ID's of the drifted clients
    :param _epochs: The number of epochs to train
    :param _mini_batch_size: The batch size to use during training
    :return: None
    """
    # FedAU: Learning module training
    learning_model_train(_model, _dataset, _server_model_params, _epochs=_epochs)

    # FedAU: Auxiliary module training, only for drifted clients #TODO: implement: drift detection/trusted clients
    if _client_id in _drifted_client_indices:
        return auxiliary_model_train(_model, _aux_dataset, _server_model_params, _epochs=_epochs,
                                     _batch_size=_mini_batch_size)

    return None


def learning_model_train(_model: nn.Module, _dataset: DataLoader, _server_model_params: OrderedDict,
                         _epochs: int) -> None:
    """
    Trains the learning model (client side), following [2].
    :param _model: Core learning model
    :param _dataset: The dataloader containing training dataset
    :param _server_model_params: The server model parameters (weights and biases from the server)
    :param _epochs: The number of epochs to train
    :return: None
    """
    # FedAU: Learning module training
    if _server_model_params is not None:
        set_parameters(_model, _server_model_params)  # apply server weights

    train(_model, _dataset, _epochs)  # regular training step for learning module


def auxiliary_model_train(_model: nn.Module, _aux_dataset: DataLoader, _server_model_params: OrderedDict, _epochs: int,
                          _batch_size: int, _verbose: bool = False) -> OrderedDict:
    """
    Trains the auxiliary model (client side), following [2].
    Then the classifier parameters of the auxiliary model are extracted and assigned to the
    _auxiliary_classifier_parameters attribute of the given client.
    :param _model: Core learning model
    :param _aux_dataset: The dataloader containing auxiliary training dataset
    :param _server_model_params: The server model parameters (weights and biases from the server)
    :param _epochs: The number of epochs to train
    :param _batch_size: The batch size to use during training
    :param _verbose: Whether to print training progress
    :return: None
    """
    # Initialize the auxiliary model (similar to learning module architecture)
    model_type = _model.get_model_type()
    if model_type == constants.ModelTypes.CNN_MODEL:
        aux_model = CNNModel().to(DEVICE)
    elif model_type == constants.ModelTypes.CNN_CIFAR_10:
        aux_model = CNNCIFAR10().to(DEVICE)
    elif model_type == constants.ModelTypes.CNN_CIFAR_100:
        aux_model = CNNCIFAR100().to(DEVICE)
    elif model_type == constants.ModelTypes.CNN_TINY_IMAGENET:
        aux_model = CNNTinyImageNet().to(DEVICE)
    else:
        raise ValueError(f"Unsupported model type for auxiliary model training: {model_type}")

    # load server parameters to auxiliary model
    set_parameters(aux_model, _server_model_params)

    # Replace fc2 layer with a new nn.Linear of the same shape
    aux_model.fc2 = nn.Linear(aux_model.fc2.in_features, aux_model.fc2.out_features)

    # Train the auxiliary model using auxiliary data; drifted samples are re-labeled using random labels
    train(aux_model, _aux_dataset, _epochs=_epochs, verbose=_verbose)

    # get the classifier of the auxiliary module
    _, _auxiliary_classifier_params = split_to_extractor_and_classifier(aux_model, None)

    return _auxiliary_classifier_params


def rapid_train(_model: nn.Module, _dataset: DataLoader, _epochs: int, _batch_size: int,
                _verbose: bool = False) -> None:
    """
    Train a model using a manual rapid-retraining style update (RRT-like).
    [1] Y. Liu, L. Xu, X. Yuan, C. Wang, and B. Li, “The Right to be Forgotten in Federated Learning: An Efficient
    Realization with Rapid Retraining,” in Proceedings of IEEE INFOCOM 2022.

    This routine performs standard forward/backward passes to obtain gradients, then **manually updates** parameters
    using an Adam-style rule with momentum (β1, β2) and a diagonal Fisher-inspired preconditioner:
    p ← p − (m_t / (v_t + ε)) / B, where
      - m_t is a bias-corrected EMA of gradients (first moment),
      - v_t ≈ sqrt(EMA(g^4)) acts as a curvature proxy (diagonal Fisher ~ g^2, and Eq. (11) in the paper
       uses ΓΓ ⇒ (g^2)^2 = g^4, hence the sqrt).

    Notes
    -----
    - This function **does not call** optimizer.step(); updates are applied inside a `with torch.no_grad():` block.
    - Gradients from each batch are snapshotted into `all_gradients` and used to compute exponentially weighted sums
    (a naive O(t) history scan). This is illustrative and memory-heavy for long runs.
    - The routine limits to `max_batches_per_epoch = 10` per epoch.

    :param _model: The model to evaluate
    :param _dataset: The test dataset
    :param _epochs: The number of epochs to train
    :param _batch_size: The batch size to use during training
    :param _verbose: Whether to print training progress
    :return: None
    """

    def get_weighed_sum(_current_grad: torch.Tensor, _all_gradients: list[dict[nn.Parameter, torch.Tensor]],
                        _beta: float, _t: int, _param: torch.nn.Parameter, _power: int = 1) -> torch.Tensor:
        """
        Compute an exponentially weighted sum of current and past gradients for a parameter.

        The function aggregates the current gradient `current_grad` and previously stored per-batch gradients from
        `all_gradients` for the same `param`, using decay factor `beta` and optional exponent `power`.

        Formally, for i in {1..t} with weights beta^(t - i):
            sum_i beta^(t - i) * g_i^power
        where g_t := current_grad and g_i (i < t) is read from all_gradients[i-1][param].

        :param _current_grad: Gradient tensor for `param` from the current backward pass.
        :param _all_gradients: History of per-batch gradient snapshots; each element maps parameters to their saved
            gradient tensors for that batch.
        :param _beta: Exponential decay factor in (0, 1); larger values give longer memory.
        :param _t: 1-based time index for the current batch within the epoch.
        :param _param:The parameter whose gradient history is being aggregated.
        :param _power: Exponent applied elementwise to each gradient before weighting. Use `power=1`
            for first-moment–like sums; `power=4` approximates the ΓΓ term in the paper’s diagonal Fisher preconditioner
            ((g^2)^2 = g^4), before taking a square root in v_t.
        :return: The weighted sum tensor (same shape as `_param`), detached from autograd.
        """
        weighed_sum = 0
        for i in range(_t):
            if _t == i + 1:
                weighed_sum += (_beta ** (_t - (i + 1))) * (_current_grad ** _power)
            else:
                weighed_sum += (_beta ** (_t - (i + 1))) * (_all_gradients[i][_param] ** _power)

        return weighed_sum

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 0.005

    criterion = nn.CrossEntropyLoss()

    _model = _model.to(DEVICE)
    _model.train()
    for epoch in range(_epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        # this loop is added because _dataset is dictionary like and torch.from_numpy() expects only Dataloader types.
        # Also takes batches of data from the dataset and trains the model

        # Define the maximum number of batches to use per epoch
        max_batches_per_epoch = 10

        all_gradients = []

        # Limit the loop to a fixed number of batches
        for batch_idx, (_x, _y) in enumerate(_dataset):
            if batch_idx >= max_batches_per_epoch:
                break

            t = batch_idx + 1
            batch_gradients = {}

            # inputs = _x.unsqueeze(1).float()  # Ensure images are in the right format and shape to feed to the model
            inputs = _x
            labels = _y

            inputs = inputs.to(DEVICE)  # move inputs to device
            labels = labels.to(DEVICE)  # move labels to device

            # Clear gradients for each batch
            _model.zero_grad()

            # Forward pass
            outputs = _model(inputs)

            # Calculate loss
            _loss = criterion(outputs, labels)

            # Backward pass -> calculates the gradients (this step gives each parameter a gradient in param.grad)
            _loss.backward()

            # Moves the weights following an update rule, leveraging the calculated gradients. Here, this is manually
            # programmed below referring to Lie et. al.[1] instead of calling optimizer.step(). Update step applies a
            # rule that moves weights a little using the gradients (param.grad) to reduce the loss.
            for param in _model.parameters():
                if param.grad is not None:
                    # saves a copy of the existing gradients
                    batch_gradients[param] = param.grad.clone().detach()

                    # calculates new weights using the existing gradients and Lie et. al.[1]'s approximation of RRT
                    weighed_sum_beta1 = get_weighed_sum(param.grad, all_gradients, beta1, t, param)
                    weighed_sum_beta2 = get_weighed_sum(param.grad, all_gradients, beta1, t, param, 4)

                    # calculates first (m_t) and second (v_t) moment estimates
                    # m_t - exponentially weighted moving average of gradients
                    # v_t - exponentially weighted moving average of squared Hessian-diagonal terms
                    m_t = ((1 - beta1) * weighed_sum_beta1) / (1 - beta1 ** t)
                    v_t = torch.sqrt(((1 - beta2) * weighed_sum_beta2) / (1 - beta2 ** t))

                    with torch.no_grad():
                        param -= (m_t / (v_t + epsilon)) / _batch_size
                    # param.data -= ((m_t) / (v_t + epsilon)) / batch_size

            all_gradients.append(batch_gradients)

            # Metrics
            epoch_loss += _loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        epoch_loss /= len(_dataset)
        epoch_acc = correct / total

        if _verbose:
            print(f"Train Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def train(_model: nn.Module, _dataset: DataLoader, _epochs: int, verbose=False) -> None:
    """
    Train the network on the training set.
    :param _model: The model to train
    :param _dataset: The dataloader containing training dataset
    :param _epochs: The number of epochs to train for
    :param verbose: Whether to print training progress
    :return: None
    """
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    _optimizer = torch.optim.Adam(_model.parameters(), lr=0.001)

    _model = _model.to(DEVICE)
    _model.train()

    for epoch in range(_epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        # this loop is added because _dataset is dictionary like and torch.from_numpy() expects only Dataloader types.
        # Also takes batches of data from the dataset and trains the model

        # Define the maximum number of batches to use per epoch
        max_batches_per_epoch = 10

        # Limit the loop to a fixed number of batches
        for batch_idx, (_x, _y) in enumerate(_dataset):
            if batch_idx >= max_batches_per_epoch:
                break

            # inputs = _x.unsqueeze(1).float()  # Ensure images are in the right format and shape to feed to the model
            inputs = _x
            labels = _y

            inputs = inputs.to(DEVICE)  # move inputs to device
            labels = labels.to(DEVICE)  # move labels to device

            # Clear gradients for each batch
            _optimizer.zero_grad()

            # Forward pass
            outputs = _model(inputs)

            # Calculate loss
            try:
                _loss = criterion(outputs, labels)
            except Exception as e:
                print(f"Error calculating loss: {e}")
                continue

            # Backward pass and optimization
            _loss.backward()
            _optimizer.step()

            # Metrics
            epoch_loss += _loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        epoch_loss /= len(_dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Train Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(_model: nn.Module, _dataset: DataLoader) -> tuple[float, float]:
    """
    Evaluate the network on the entire test set.
    :param _model: The model to evaluate
    :param _dataset: The test dataset
    :return: Tuple of loss and accuracy
    """
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0

    _model = _model.to(DEVICE)
    _model.eval()

    with torch.no_grad():
        # this loop is added because _dataset is dictionary like and torch.from_numpy() expects only Dataloader types
        for _x, _y in _dataset:
            # inputs = _x.unsqueeze(1).float()   # Ensure images are in the right format and shape to feed to the model
            inputs = _x
            labels = _y

            inputs = inputs.to(DEVICE)  # move inputs to device
            labels = labels.to(DEVICE)  # move labels to device

            # forward pass
            outputs = _model(inputs)

            # compute loss
            loss += criterion(outputs, labels).item()

            # compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # average loss over all samples
    loss /= len(_dataset)
    accuracy = correct / total
    return loss, accuracy


def set_parameters(_model: nn.Module, parameters: OrderedDict, _strict=True) -> None:
    """
    Set the model weights and biases
    :param _model: The model to set parameters for
    :param parameters: The parameters to set
    :param _strict: Whether to strictly enforce that the keys in state_dict match the keys returned by the model's
    :return: None
    """
    _, _ = _model.load_state_dict(parameters, strict=_strict)


def set_parameters_ema(_model: nn.Module, parameters: OrderedDict, alpha: float, _strict: bool = True) -> None:
    """
    Set model parameters using EMA update:
        new = alpha * old + (1 - alpha) * incoming

    :param _model: Model whose parameters will be updated
    :param parameters: Incoming parameters (state_dict-like OrderedDict)
    :param alpha: EMA coefficient (0 <= alpha <= 1)
    :param _strict: Whether to strictly enforce key matching
    """

    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    current_state = _model.state_dict()
    ema_state = OrderedDict()

    for key in current_state.keys():
        if key not in parameters:
            if _strict:
                raise KeyError(f"Key '{key}' missing in incoming parameters")
            # keep old parameter if not present
            ema_state[key] = current_state[key]
            continue

        # Ensure tensors are on same device / dtype
        old_param = current_state[key]
        new_param = parameters[key].to(old_param.device)

        # EMA update
        ema_state[key] = alpha * old_param + (1.0 - alpha) * new_param

    # Load EMA-updated parameters
    _, _ = _model.load_state_dict(ema_state, strict=_strict)


def get_parameters_as_np_array(_model: nn.Module) -> List[np.ndarray]:
    """
    Set the model weights and biases
    :param _model: The model to get parameters from
    :return: List of model parameters as numpy arrays
    """
    return [val.cpu().numpy() for _, val in _model.state_dict().items()]
