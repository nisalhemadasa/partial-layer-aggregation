"""
Description: This module defines feedforward neural network models for image classification using PyTorch.

Author: Nisal Hemadasa
Date: 04-04-2025
Version: 2.0
"""
from typing import Tuple, OrderedDict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__}"
)


class SimpleModel(nn.Module):
    def __init__(self):
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


class CNNModel(nn.Module):
    def __init__(self):
        """Defined for both MNIST and F_MNIST datasets"""
        super(CNNModel, self).__init__()

        self.pool = nn.MaxPool2d(2, 2, 1, 2)

        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 2)
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 2)
        self.conv4 = nn.Conv2d(32, 64, 3, 1, 2)

        self.fc1 = nn.Linear(64 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm1d(120)
        self.bn6 = nn.BatchNorm1d(84)

    def forward(self, x):
        # print("Input shape:", x.shape)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # print("After conv1 and pool:", x.shape)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # print("After conv2 and pool:", x.shape)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # print("After conv3 and pool:", x.shape)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # print("After conv4 and pool:", x.shape)
        x = x.view(-1, 64 * 4 * 4)
        # print("After view:", x.shape)
        x = F.relu(self.bn5(self.fc1(x)))
        # print("After fc1:", x.shape)
        x = F.relu(self.bn6(self.fc2(x)))
        # print("After fc2:", x.shape)
        x = self.fc3(x)  # classifier in the case of FedAU
        # print("Output shape:", x.shape)
        return x


def split_to_extractor_and_classifier(_model: nn.Module, _model_params: OrderedDict) -> tuple[OrderedDict, OrderedDict]:
    """
    Split the model/model parameters into feature extractor and classifier parts. The feature extractor includes all layers except the
    final fully connected layer (fc3), while the classifier includes only the final fully connected layer.
    :param _model: The model to split
    :param _model_params: The model parameters to split
    :return: A tuple containing two OrderedDicts: (parameters of the feature extractor, parameters of the classifier)
    """
    if _model is not None:
        _model_params = _model.state_dict()

    # get the extractor parameters (all except fc3 layer)
    extractor_params = OrderedDict((k, v) for k, v in _model_params.items() if not k.startswith("fc3."))

    # get the classifier parameters (fc3 layer)
    classifier_params = OrderedDict((k, v) for k, v in _model_params.items() if k.startswith("fc3."))

    return extractor_params, classifier_params


def fedau_clientside_train(_model: nn.Module, _dataset: DataLoader, _server_model_params: OrderedDict,
                           _drifted_client_indices: List[int], _client_id, _epochs: int,
                           _mini_batch_size: int) -> OrderedDict:
    """
    Performs clients dei training operations of the FedAU algorithm, following [2]. 
    This includes training the (1) learning module and (2) auxiliary module
    
    [2] Implementation of FedAU algorithm following the paper: [2]H. Gu, G. Zhu, J. Zhang, X. Zhao, Y. Han, L. Fan and
    Q. Yang, “Unlearning during Learning: An Efficient Federated Machine Unlearning Method,” in
    Proceedings of the 33rd International Joint Conference on Artificial Intelligence (IJCAI-24)
    :param _model: Core learning model
    :param _dataset: The dataloader containing training dataset
    :param _server_model_params: The server model parameters (weights and biases from the server)
    :param _client_id: ID of the given client
    :param _drifted_client_indices: List of ID's of the drifted clients
    :param _epochs: The number of epochs to train
    :param _mini_batch_size: The batch size to use during training
    :return: None
    """
    # FedAU: Learning module training
    learning_model_train(_model, _dataset, _server_model_params, _epochs=_epochs, )

    # FedAU: Auxiliary module training, only for drifted clients #TODO: implement: drift detection/trusted clients
    if _client_id in _drifted_client_indices:
        return auxiliary_model_train(_model, _dataset, _server_model_params, _epochs=_epochs,
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


def auxiliary_model_train(_model: nn.Module, _dataset: DataLoader, _server_model_params: OrderedDict, _epochs: int,
                          _batch_size: int, _verbose: bool = False) -> OrderedDict:
    """
    Trains the auxiliary model (client side), following [2].
    Then the classifier parameters of the auxiliary model are extracted and assigned to the
    _auxiliary_classifier_parameters attribute of the given client.
    :param _model: Core learning model
    :param _dataset: The dataloader containing training dataset
    :param _server_model_params: The server model parameters (weights and biases from the server)
    :param _epochs: The number of epochs to train
    :param _batch_size: The batch size to use during training
    :param _verbose: Whether to print training progress
    :return: None
    """
    # Initialize the auxiliary model
    aux_model = CNNModel().to(DEVICE)  # initialize using a fresh base model, similar to learning module architecture
    set_parameters(aux_model, _server_model_params) # load server parameters to auxiliary model

    # Replace fc3 layer with a new nn.Linear of the same shape
    aux_model.fc3 = nn.Linear(aux_model.fc3.in_features, aux_model.fc3.out_features)

    # Train the auxiliary model using data with new patterns
    # TODO: create trainloader for drifted local_trainset and non-drifted local_trainset
    # drifted_dataset =
    # non - drifted_dataset =

    train(aux_model, _dataset, _epochs=_epochs, verbose=_verbose)

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
            _loss = criterion(outputs, labels)

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
    _model.eval()
    with torch.no_grad():
        # this loop is added because _dataset is dictionary like and torch.from_numpy() expects only Dataloader types
        for _x, _y in _dataset:
            # inputs = _x.unsqueeze(1).float()   # Ensure images are in the right format and shape to feed to the model
            inputs = _x
            labels = _y

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


def set_parameters(_model: nn.Module, parameters: OrderedDict) -> None:
    """
    Set the model weights and biases
    :param _model: The model to set parameters for
    :param parameters: The parameters to set
    :return: None
    """
    _model.load_state_dict(parameters, strict=True)


def get_parameters_as_np_array(_model:  nn.Module) -> List[np.ndarray]:
    """
    Set the model weights and biases
    :param _model: The model to get parameters from
    :return: List of model parameters as numpy arrays
    """
    return [val.cpu().numpy() for _, val in _model.state_dict().items()]
