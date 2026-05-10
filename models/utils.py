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
from models import SimpleModel, CNNModel, CNNCIFAR10, CNNCIFAR100, TabularAdultModel, ConvNeXtTinyImageNet
from models.CNNCIFAR100.model import ResNet18CIFAR100, ShallowResNetCIFAR100

DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__}"
)


def split_to_extractor_and_classifier(_model: nn.Module, _model_params: OrderedDict, model_type: str) -> tuple[
    OrderedDict, OrderedDict]:
    """
    Split the model/model parameters into feature extractor and classifier parts. The feature extractor includes all layers except the
    final fully connected layer (fc2), while the classifier includes only the final fully connected layer.
    :param _model: The model to split
    :param _model_params: The model parameters to split
    :return: A tuple containing two OrderedDicts: (parameters of the feature extractor, parameters of the classifier)
    """
    if _model is not None:
        _model_params = _model.state_dict()

    # get the extractor parameters (all except fc2 and fc1 layer)
    if model_type is not constants.ModelTypes.CONVNET_TINY_IMAGENET:
        # For Deep CNNS
        classifier_params = OrderedDict((k, v) for k, v in _model_params.items() if
                                        k.startswith("fc1.") or k.startswith("fc2.") or k.startswith("fc3.") or k.startswith("fc4."))
        # classifier_params = OrderedDict((k, v) for k, v in _model_params.items() if
        #                                 k.startswith("fc4.") )

        # For simple CNNs
        # extractor_params = OrderedDict((k, v) for k, v in _model_params.items() if
        #                                not (k.startswith("fc1.") or k.startswith("fc2.") or k.startswith('conv2')))

        # For ResNet like
        extractor_params = OrderedDict((k, v) for k, v in _model_params.items() if
                                       not (k.startswith("fc1.") or k.startswith("fc2.") or
                                            k.startswith("layer4.") or k.startswith("layer3.")))
    else:
        # For Tiny ImageNet200 model
        extractor_params = OrderedDict((k, v) for k, v in _model_params.items() if
                                       not (k.startswith("head.")))  # extractor layers start with as 'norm.'

    if model_type is not constants.ModelTypes.CONVNET_TINY_IMAGENET:
        # get the classifier parameters (fc2 layer)

        # For Deep CNNS
        classifier_params = OrderedDict((k, v) for k, v in _model_params.items() if
                                        k.startswith("fc1.") or k.startswith("fc2.") or k.startswith("fc3.") or k.startswith("fc4."))
        # classifier_params = OrderedDict((k, v) for k, v in _model_params.items() if
        #                                 k.startswith("fc4.") )

        # For ResNet like
        # classifier_params = OrderedDict((k, v) for k, v in _model_params.items() if
        #                                 k.startswith("fc2.") or k.startswith("layer4.") or k.startswith("layer3."))

        # For simple CNN
        # classifier_params = OrderedDict((k, v) for k, v in _model_params.items() if
        #                                 (k.startswith("fc1.") or k.startswith("fc2.")))
    else:
        # For Tiny ImageNet200 model
        classifier_params = OrderedDict((k, v) for k, v in _model_params.items() if k.startswith("head."))

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
        # aux_model = ShallowResNetCIFAR100().to(DEVICE)
        aux_model = ResNet18CIFAR100().to(DEVICE)
        # aux_model = CNNCIFAR100().to(DEVICE)
    elif model_type == constants.ModelTypes.CNN_TINY_IMAGENET:
        aux_model = ConvNeXtTinyImageNet.to(DEVICE)
    elif model_type == constants.ModelTypes.TABULAR_ADULT:
        # New branch for TabularAdultModel
        aux_model = TabularAdultModel().to(DEVICE)
    else:
        raise ValueError(f"Unsupported model type for auxiliary model training: {model_type}")

    # load server parameters to auxiliary model
    set_parameters(aux_model, _server_model_params)

    # Replace fc2 layer with a new nn.Linear of the same shape
    aux_model.fc2 = nn.Linear(aux_model.fc2.in_features, aux_model.fc2.out_features)

    # Train the auxiliary model using auxiliary data; drifted samples are re-labeled using random labels
    train(aux_model, _aux_dataset, _epochs=_epochs, verbose=_verbose)

    # get the classifier of the auxiliary module
    _, _auxiliary_classifier_params = split_to_extractor_and_classifier(aux_model, None, aux_model.get_model_type())

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


def train(_model: nn.Module, _dataset: DataLoader, _epochs: int, verbose: bool = False) -> None:
    """
    Train the network on the training set.
    :param _model: The model to train
    :param _dataset: The dataloader containing training dataset
    :param _epochs: The number of epochs to train for
    :param verbose: Whether to print training progress
    :return: None
    """
    # # For generic trainng
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    #
    # _model = _model.to(DEVICE)
    #
    # optimizer = torch.optim.AdamW(
    #     _model.parameters(),
    #     lr=3e-4,
    #     weight_decay=5e-2
    # )
    #
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=_epochs,
    #     eta_min=1e-6
    # )
    #
    # use_amp = torch.cuda.is_available()
    # scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # For CIFAR-10 Boosting
    criterion = nn.NLLLoss()
    _model = _model.to(DEVICE).float()

    optimizer = torch.optim.SGD(
        _model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 120, 160],
        gamma=0.2
    )

    use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    for epoch in range(_epochs):
        _model.train()

        correct = 0
        total = 0
        epoch_loss = 0.0
        num_batches = 0

        for _x, _y in _dataset:
            inputs = _x.to(DEVICE, non_blocking=True).float()
            labels = _y.to(DEVICE, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = _model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        scheduler.step()

        epoch_loss /= num_batches
        epoch_acc = correct / total

        if verbose:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Train Epoch {epoch + 1}: "
                f"train loss {epoch_loss:.4f}, "
                f"accuracy {epoch_acc:.4f}, "
                f"lr {current_lr:.6f}"
            )


# def train(_model: nn.Module, _dataset: DataLoader, _epochs: int, verbose=False) -> None:
#     """
#     Train the network on the training set.
#     :param _model: The model to train
#     :param _dataset: The dataloader containing training dataset
#     :param _epochs: The number of epochs to train for
#     :param verbose: Whether to print training progress
#     :return: None
#     """
#     # criterion = nn.BCEWithLogitsLoss()
#     # criterion = nn.BCELoss()
#     criterion = nn.CrossEntropyLoss()
#     _optimizer = torch.optim.Adam(_model.parameters(), lr=0.001)
#
#     _model = _model.to(DEVICE)
#     _model.train()
#
#     for epoch in range(_epochs):
#         correct, total, epoch_loss = 0, 0, 0.0
#         # this loop is added because _dataset is dictionary like and torch.from_numpy() expects only Dataloader types.
#         # Also takes batches of data from the dataset and trains the model
#
#         # Define the maximum number of batches to use per epoch
#         max_batches_per_epoch = 10
#
#         # Limit the loop to a fixed number of batches
#         for batch_idx, (_x, _y) in enumerate(_dataset):
#             if batch_idx >= max_batches_per_epoch:
#                 break
#
#             # inputs = _x.unsqueeze(1).float()  # Ensure images are in the right format and shape to feed to the model
#             inputs = _x
#             labels = _y
#
#             inputs = inputs.to(DEVICE)  # move inputs to device
#             labels = labels.to(DEVICE)  # move labels to device
#
#             # Clear gradients for each batch
#             _optimizer.zero_grad()
#
#             # Forward pass
#             outputs = _model(inputs)
#
#             # Calculate loss
#             try:
#                 _loss = criterion(outputs, labels)
#             except Exception as e:
#                 print(f"Error calculating loss: {e}")
#                 continue
#
#             # Backward pass and optimization
#             _loss.backward()
#             _optimizer.step()
#
#             # Metrics
#             epoch_loss += _loss.item()
#             total += labels.size(0)
#             correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
#
#         epoch_loss /= len(_dataset)
#         epoch_acc = correct / total
#         if verbose:
#             print(f"Train Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")


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
