"""
Description: This module defines a server of the federated network.

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
from typing import List, OrderedDict, Tuple, Dict, Any

from torch import Tensor
from torch.utils.data import DataLoader

import constants
import strategy
from distance_metrics.distance_metrics import compute_euclidean_distance_weights
from drift_concepts.drift import Drift
from federated_network.client import DEVICE, Client
from models.model import SimpleModel, CNNModel, test


class Server:
    def __init__(self, _server_id, _abs_id, _strategy, _model, _client_ids=None):
        self.server_id = _server_id
        self.abs_id = _abs_id  # Absolute ID that keeps a running count of the servers in the server hierarchy
        self.strategy = _strategy
        self.model = _model
        self.client_ids = []  # List of client IDs the server is connected to in the federated network
        self.child_server_ids = []  # List of child server IDs in the server hierarchy
        self.parent_server_id = None  # Parent server ID in the server hierarchy

    def train(self, client_model_parameters: Dict[str, OrderedDict],
              aux_classifier_parameters: Dict[str, OrderedDict] = None, ema_weight: float = None) -> None:
        """
        Train the server model using the client model parameters.
        :param client_model_parameters: List of client model parameters
        :param aux_classifier_parameters: List of auxiliary classifier parameters from drifted clients
        :param ema_weight: EMA weight parameter for FedAU algorithm
        :return: None
        """
        if aux_classifier_parameters:
            # check if all the elements in the aux_classifier_parameters are None
            if all(param is None for param in aux_classifier_parameters.values()):
                return
            # FedAU: For servers with clients with auxiliary classifiers (i.e., leaf servers, having drifted nodes)
            self.strategy.aggregate_models(self.model, client_model_parameters, aux_classifier_parameters,
                                           ema_weight)
        else:
            # FedAvg: For internal servers or when there are no drifted clients
            self.strategy.aggregate_models(self.model, client_model_parameters)

    def evaluate(self, _test_set: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the server model using the validation data.
        :param _test_set: test data
        :return: loss and accuracy
        """
        loss, accuracy = test(self.model, _test_set)
        return float(loss), float(accuracy)


def model_aggregation(server_hierarchy: List[List[Server]], server_test_set: DataLoader, sampled_clients: List[Client],
                      drift: Drift, fedau_alpha: float, is_evaluate_server_model=False, verbose=False) -> List:
    """
    Aggregate the models of the clients to the server model.
    :param server_hierarchy: List of servers in the hierarchy
    :param server_test_set: List of test data for server model evaluation, once the aggregation is done
    :param sampled_clients: List of sampled clients
    :param drift: Drift instance
    :param fedau_alpha: EMA weight (alpha) parameter for the FedAU algorithm
    :param is_evaluate_server_model: Boolean flag to indicate whether to evaluate the server model during aggregation
    :param verbose: Whether to print detailed logs or not
    :return: List of loss and accuracy at each level of the server hierarchy; outer list: server hierarchy levels,
    inner list: loss and accuracy Tuple at each level (loss, accuracy)
    """
    # Store the loss and accuracy at each level of the server model hierarchy
    server_loss_and_accuracy = []

    # EMA weight, and auxiliary classifier parameters, for the FedAU algorithm
    ema_weight = None
    client_aux_classifier_parameters = None

    # Evaluate server model on the upward traversal aggregation  only if the hierarchy has more than one level. Else,
    # this evaluation will be redundant as it is already done during the server model distribution phase.
    if len(server_hierarchy) > 1:
        is_evaluate_server_model = True

    if verbose:
        print('aggregate client models')

    # Aggregate the models of the clients to the server model.Start by aggregating the leaves and move up the hierarchy
    for depth_level in range(len(server_hierarchy) - 1, -1, -1):
        loss_and_accuracy_at_level = []

        for server in server_hierarchy[depth_level]:
            if depth_level == len(server_hierarchy) - 1:
                # Leaf nodes
                # Get client (learning) model parameters
                client_model_parameters = {client_id: sampled_clients[client_id].model.state_dict()
                                           for client_id in server.client_ids}

                # TODO: beware that FedAU depicted behavior of performance compromise, that this code section may cause
                # Get auxiliary classifier parameters from drifted clients only
                if drift.is_drift:
                    if server.strategy == constants.RecoveryAlgorithm.FEDAU or server.strategy == constants.RecoveryAlgorithm.FLUID:
                        print('inside!!!')
                        drifted_client_ids = set(drift.drifted_client_indices or [])
                        if drift.is_drift and drifted_client_ids:
                            # Collect the parameters to a dictionary (client_id: aux_classifier_parameters)
                            client_aux_classifier_parameters = {client_id:
                                                                    (sampled_clients[
                                                                         client_id].auxiliary_classifier_parameters  # get aux classifier params
                                                                     if client_id in drifted_client_ids else None)
                                                                # drifted clients only
                                                                for client_id in
                                                                server.client_ids}  # connected to this server

                            ema_weight = fedau_alpha

                if verbose:
                    print('server:' + str(server.server_id) + ' -> ' + 'clients:' + str(server.client_ids))

                # Aggregate client models
                server.train(client_model_parameters, client_aux_classifier_parameters, ema_weight)
            else:
                # Internal nodes: Aggregate models from child servers
                # Collect the parameters to a dictionary (server_id: server_model_parameters)
                child_server_model_parameters = {
                    child_server: server_hierarchy[depth_level + 1][child_server].model.state_dict()
                    for child_server in server.child_server_ids}

                # Aggregate child server models. Auxiliary classifier parameters are not used in internal server nodes
                # since they are not connected to clients which train auxiliary modules
                server.train(child_server_model_parameters, None, ema_weight)

            # Evaluate the server model
            if is_evaluate_server_model:
                loss, accuracy = server.evaluate(server_test_set)
                loss_and_accuracy_at_level.append((loss, accuracy))

        if is_evaluate_server_model:
            server_loss_and_accuracy.append(loss_and_accuracy_at_level)

    if is_evaluate_server_model:
        server_loss_and_accuracy.reverse()  # Reverse the list to get the root first, to be consistent throughout the code

    return server_loss_and_accuracy


def model_distribution(server_hierarchy: List[List[Server]], server_test_set: DataLoader) -> List:
    """
    The aggregated models are distributed down the hierarchy. I.e., the edge models are updated by the global model and
    client models are updated by the edge models.
    :param server_hierarchy: List of servers in the hierarchy
    :param server_test_set: List of test data for server model evaluation, once the aggregation is done
    :return: None
    """
    # Store the loss and accuracy at each level of the server model hierarchy
    server_loss_and_accuracy = []
    global_avg_loss_and_accuracy = None

    # Evaluate the accuracy of the root server model
    global_server = server_hierarchy[0][0]
    loss, accuracy = global_server.evaluate(server_test_set)

    # Store the loss and accuracy of the global server model
    server_loss_and_accuracy.append([(loss, accuracy)])

    # Aggregate the global server to the edge server down the hierarchy starting from the leaf nodes
    for depth_level in range(len(server_hierarchy) - 1):
        loss_and_accuracy_at_level = []

        # Update the edge server model with the global server model
        for server in server_hierarchy[depth_level + 1]:
            # Get global server parameters and update the edge server model
            parent_server = server_hierarchy[depth_level][server.parent_server_id]
            server_parameters = {server.server_id: server.model.state_dict(),
                                 parent_server.server_id: parent_server.model.state_dict()}
            # Update edge server models
            server.train(server_parameters, None, None)

            # Evaluate the edge server model
            loss, accuracy = server.evaluate(server_test_set)
            loss_and_accuracy_at_level.append((loss, accuracy))

        server_loss_and_accuracy.append(loss_and_accuracy_at_level)

    return server_loss_and_accuracy


def change_server_aggregation_strategy(server_hierarchy: List[Any], drift_recovery_method: str, drift: Drift) -> None:
    """
    Change the aggregation strategy of the leaf servers (only) of the hierarchy.
    :param server_hierarchy: List of servers in the hierarchy
    :param drift_recovery_method: Drift recovery method
    :param drift: Drift instance
    :return: None
    """
    if drift_recovery_method == constants.RecoveryAlgorithm.FEDAU or drift_recovery_method == constants.RecoveryAlgorithm.FLUID:
        for server in server_hierarchy[-1]:  # applied only to leaf servers
            # change the strategy only in the servers where drifted clients are connected
            drifted = set(drift.drifted_client_indices or [])  # makes sure it's at least an empty set and not None
            if set(server.client_ids) & drifted:  # checks if there is any intersection
                server.strategy = strategy.FedAU.aggregator_fn()
    else:
        # if the drift is ended, change the strategy back to FedAvg
        for server in server_hierarchy[-1]:
            server.strategy = strategy.FedAvg.aggregator_fn()


def server_fn(server_id: int, dataset_name: str, server_abs_id: int) -> Server:
    """
    Create a server instances on demand for the optimal use of resources.
    :param server_id: Server ID
    :param dataset_name: Name of the dataset
    :param server_abs_id: Absolute server ID; a running count of all the servers created
    :returns Server: A Server instance.
    """
    aggregator_strategy = strategy.FedAvg.aggregator_fn()
    # model = SimpleModel().to(DEVICE)
    if dataset_name == constants.DatasetNames.CIFAR_10:
        model = CNNModel().to(DEVICE)
    else:
        # MNIST
        model = CNNModel().to(DEVICE)

    return Server(_server_id=server_id, _abs_id=server_abs_id, _strategy=aggregator_strategy, _model=model)
