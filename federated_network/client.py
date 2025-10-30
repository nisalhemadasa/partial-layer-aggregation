"""
Description: This module defines a client of the federated network.

Author: Nisal Hemadasa
Date: 18-10-2024
Version: 1.0
"""
import importlib
import random
from collections import OrderedDict
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

import constants
from data.utils import convert_dataset_to_loader
from drift_concepts.drift import Drift
from models.model import train, test, CNNModel, rapid_train, auxiliary_model_train

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__}"
)


class Client:
    def __init__(self, client_id, if_iid, model, epochs, mini_batch_size, local_trainset, testset):
        self.client_id = client_id
        self.iid = if_iid  # whether the client has IID data or not
        self.model = model
        self.epochs = epochs
        self.local_trainset = local_trainset
        self.testset = testset
        self.mini_batch_size = mini_batch_size
        self.trainloader = None  # initialized only when sample_data() is called
        self.testloader = None  # initialized only when sample_data() is called
        self.parent_server_id = None  # server ID in the server hierarchy to which the client is connected
        self.auxiliary_classifier_parameters = None  # distance of the client from the server in the server hierarchy

    def get_model_weights(self):
        """ Get the model weights and biases """
        return self.model.state_dict()

    def sample_data(self):
        """ Sample data from the train and test datasets unique to each client and create DataLoaders"""
        # Create a DataLoader using a randomly sampled subset(fraction_of_data%) from the local training data
        fraction_of_data = 0.1
        subset_size = int(len(self.local_trainset) * fraction_of_data)
        indices = random.sample(range(len(self.local_trainset)), subset_size)
        subset = Subset(self.local_trainset, indices)

        self.trainloader = convert_dataset_to_loader(_dataset=subset,
                                                     _batch_size=self.mini_batch_size)
        self.testloader = convert_dataset_to_loader(_dataset=self.testset, _batch_size=self.mini_batch_size,
                                                    _is_shuffle=False)

    def fit(self, server_model_parameters: OrderedDict, drift_recovery_method: str, drift: Drift, _client_id: int):
        """ Train the client model using new data and server parameters and return the updated model weights and
        biases"""
        if not drift.is_drift:
            # Do not set the server weights and biases if the server aggregation is not None (e.g. initial round)
            if server_model_parameters is not None:
                set_parameters(self.model, server_model_parameters)  # apply server weights
            # Train the client model using new data and server parameters
            train(self.model, self.trainloader, self.epochs)
        else:
            # Train the client model using new data and server parameters
            if drift_recovery_method == constants.RecoveryAlgorithm.ADAM_BASED:
                # Adam-based recovery (1st order) + reinitialization of client parameters from the global model from scratch
                train(self.model, self.trainloader, _epochs=self.epochs)
            elif drift_recovery_method == constants.RecoveryAlgorithm.RRT:
                # Rapid retraining (2nd order) + reinitialization of client parameters from the global model from scratch
                rapid_train(self.model, self.trainloader, _epochs=self.epochs, _batch_size=self.mini_batch_size)
            elif drift_recovery_method == constants.RecoveryAlgorithm.FEDAU:
                # FedAU client side operations
                fedau_clientside_train(self.model, self.trainloader, server_model_parameters,
                                       _drifted_client_indices, self.auxiliary_classifier_parameters, _client_id,
                                       _epochs=self.epochs, _mini_batch_size=self.mini_batch_size)

    def evaluate(self):
        """ Evaluate the client model on the validation data and return the loss and accuracy """
        loss, accuracy = test(self.model, self.testloader)
        return float(loss), float(accuracy)


def set_parameters(_model, parameters: OrderedDict):
    """ Set the model weights and biases """
    _model.load_state_dict(parameters, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    """ Set the model weights and biases """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def client_initial_training(_clients: List[Client]) -> List:
    """
    Train the clients initially using their local data.
    :param _clients: List of client instances
    :return:  List of loss and accuracy of each client after the initial training
    """
    initial_client_loss_and_accuracy = []
    # All the clients are trained individually using local data initially
    for client in _clients:
        client.sample_data()
        client.fit(None)
        initial_client_loss_and_accuracy.append(client.evaluate())

    return initial_client_loss_and_accuracy


def client_fn(client_id: int, if_iid: bool, num_local_epochs: int, mini_batch_size: int,
              _dataset: List[Dataset]) -> Client:
    """
    Create a client instances on demand for the optimal use of resources.
    :param client_id: client id
    :param if_iid: whether the client has IID data or not
    :param num_local_epochs: number of local epochs, before being aggregation ready
    :param mini_batch_size: size of the batches for the clients to train on
    :param _dataset: train and test datasets
    :returns Client: A Client instance.
    """
    # Load model
    _model = CNNModel()

    # Unpacking _dataset (which contains a subset of the complete training set (e.g., MNIST) and the global test set)
    local_trainset, testset = _dataset

    # Create a  single Flower client representing a single organization
    return Client(client_id=client_id, if_iid=if_iid, model=_model, epochs=num_local_epochs,
                  mini_batch_size=mini_batch_size, local_trainset=local_trainset, testset=testset)
