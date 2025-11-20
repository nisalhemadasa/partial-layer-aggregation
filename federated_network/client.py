"""
Description: This module defines a client of the federated network.

Author: Nisal Hemadasa
Date: 18-10-2024
Version: 1.0
"""
import random
from collections import OrderedDict
from typing import List

import torch
from torch.utils.data import Dataset, Subset, DataLoader

import constants
from data.utils import convert_dataset_to_loader
from models.model import train, test, CNNModel, rapid_train, fedau_clientside_train, set_parameters

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__}"
)


class Client:
    def __init__(self, client_id, if_iid, model, epochs, mini_batch_size, local_trainset, testset,
                 drift_recovery_method):
        self.client_id = client_id
        self.iid = if_iid  # whether the client has IID data or not
        self.model = model
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.local_trainset = local_trainset
        self.testset = testset
        self.drift_recovery_method = drift_recovery_method
        self.trainloader = None  # initialized only when sample_data() is called
        self.testloader = None  # initialized only when sample_data() is called
        self.parent_server_id = None  # server ID in the server hierarchy to which the client is connected
        self.auxiliary_classifier_parameters = None  # distance of the client from the server in the server hierarchy
        self.aux_dataloader = None  # dateset with random labels for training the auxiliary classifier in FedAU

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

    def fit(self, _is_drift: bool, _is_drift_end: bool, server_model_parameters: OrderedDict, _client_id: int,
            drift_recovery_method: str, _drifted_client_indices: List[int]) -> None:
        """
        Train the client model using new data and server parameters and return the updated model weights and
        biases
        :param _is_drift: Flag indicating whether drift has occurred or not
        :param _is_drift_end: Flag indicating whether the drift period has ended or not
        :param server_model_parameters: Aggregated server model parameters
        :param drift_recovery_method: Drift recovery method to be used
        :param _drifted_client_indices: List of client indices that have experienced drift
        :param _client_id: Client ID of the given client
        :return: None
        """
        if not _is_drift:
            if _is_drift_end and drift_recovery_method == constants.RecoveryAlgorithm.FLUID:  # after drift ends
                # Rapid retraining (2nd order) + reinitialization of client parameters from the global model from scratch
                rapid_train(self.model, self.trainloader, _epochs=self.epochs, _batch_size=self.mini_batch_size)
            else:  # before drift begins
                # Train the client model using new data and server parameters
                train(self.model, self.trainloader, self.epochs)
        else:
            # Train the client model using new data and server parameters
            if drift_recovery_method == constants.RecoveryAlgorithm.FEDAVG:
                # Adam-based recovery (1st order) + reinitialization of client parameters from the global model from scratch
                train(self.model, self.trainloader, _epochs=self.epochs)
            elif drift_recovery_method == constants.RecoveryAlgorithm.RRT:
                # Rapid retraining (2nd order) + reinitialization of client parameters from the global model from scratch
                rapid_train(self.model, self.trainloader, _epochs=self.epochs, _batch_size=self.mini_batch_size)
            elif drift_recovery_method == constants.RecoveryAlgorithm.FEDAU or drift_recovery_method == constants.RecoveryAlgorithm.FLUID:
                # FedAU client side operations
                self.auxiliary_classifier_parameters = fedau_clientside_train(self.model, self.trainloader,
                                                                              self.aux_dataloader,
                                                                              server_model_parameters,
                                                                              _drifted_client_indices, _client_id,
                                                                              _epochs=self.epochs,
                                                                              _mini_batch_size=self.mini_batch_size)

    def evaluate(self):
        """ Evaluate the client model on the validation data and return the loss and accuracy """
        loss, accuracy = test(self.model, self.testloader)
        return float(loss), float(accuracy)


def client_initial_training(_clients: List[Client], _is_drift: bool, _is_drift_end: bool) -> List:
    """
    Train the clients initially using their local data.
    :param _clients: List of client instances
    :param _is_drift: Flag indicating whether drift has occurred or not
    :param _is_drift_end: Flag indicating whether the drift period has ended or not
    :return:  List of loss and accuracy of each client after the initial training
    """
    initial_client_loss_and_accuracy = []
    # All the clients are trained individually using local data initially
    for client in _clients:
        client.sample_data()
        # We assume no drift during initial training. Hence, drift related parameters are set to None
        client.fit(_is_drift, _is_drift_end, None, client.client_id, None, None)
        initial_client_loss_and_accuracy.append(client.evaluate())

    return initial_client_loss_and_accuracy


def change_client_drift_recovery_method(clients: List[Client], drift_recovery_method: str,
                                        drifted_client_indices: List[int]) -> None:
    """
    Change the drift recovery method of the clients.
    :param clients: List of client instances
    :param drift_recovery_method: New drift recovery method to be set
    :param drifted_client_indices: List of client indices that have experienced drift
    """
    for client in clients:
        if client.client_id in drifted_client_indices:
            client.drift_recovery_method = drift_recovery_method


def client_fn(client_id: int, if_iid: bool, num_local_epochs: int, mini_batch_size: int,
              _dataset: List[Dataset], drift_recovery_method: str) -> Client:
    """
    Create a client instances on demand for the optimal use of resources.
    :param client_id: client id
    :param if_iid: whether the client has IID data or not
    :param num_local_epochs: number of local epochs, before being aggregation ready
    :param mini_batch_size: size of the batches for the clients to train on
    :param _dataset: train and test datasets
    :param drift_recovery_method: Drift recovery method to be used by the client
    :returns Client: A Client instance.
    """
    # Load model
    _model = CNNModel()

    # Unpacking _dataset (which contains a subset of the complete training set (e.g., MNIST) and the global test set)
    local_trainset, testset = _dataset

    # Create a  single Flower client representing a single organization
    return Client(client_id=client_id, if_iid=if_iid, model=_model, epochs=num_local_epochs,
                  mini_batch_size=mini_batch_size, local_trainset=local_trainset, testset=testset,
                  drift_recovery_method=drift_recovery_method)
