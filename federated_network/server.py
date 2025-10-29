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

    def train(self, client_model_parameters: List[OrderedDict]) -> None:
        """
        Train the server model using the client model parameters.
        :param client_model_parameters: List of client model parameters
        :return: None
        """
        self.model = self.strategy.aggregate_models(self.model, client_model_parameters)

    def evaluate(self, _test_set: DataLoader) -> (float, float):
        """
        Evaluate the server model using the validation data.
        :param _test_set: test data
        :return: loss and accuracy
        """
        loss, accuracy = test(self.model, _test_set)
        return float(loss), float(accuracy)


def model_aggregation(server_hierarchy: List[List[Server]], sampled_clients_model_parameters: List[OrderedDict],
                      server_test_set: DataLoader, is_evaluate_server_model=False) -> List:
    """
    Aggregate the models of the clients to the server model.
    :param server_hierarchy: List of servers in the hierarchy
    :param sampled_clients_model_parameters: List of client model parameters
    :param server_test_set: List of test data for server model evaluation, once the aggregation is done
    :param is_evaluate_server_model: Boolean flag to indicate whether to evaluate the server model during aggregation
    :return: List of loss and accuracy at each level of the server hierarchy; outer list: server hierarchy levels,
    inner list: loss and accuracy Tuple at each level (loss, accuracy)
    """
    # Store the loss and accuracy at each level of the server model hierarchy
    server_loss_and_accuracy = []

    # Evaluate server model on the upward traversal aggregation  only if the hierarchy has more than one level. Else,
    # this evaluation will be redundant as it is already done during the server model distribution phase.
    if len(server_hierarchy) > 1:
        is_evaluate_server_model = True

    print('aggregate client models')

    # Aggregate the models of the clients to the server model.Start by aggregating the leaves and move up the hierarchy
    for depth_level in range(len(server_hierarchy) - 1, -1, -1):
        loss_and_accuracy_at_level = []

        for server in server_hierarchy[depth_level]:
            if depth_level == len(server_hierarchy) - 1:
                # Leaf nodes: Aggregate client models
                client_model_parameters = [sampled_clients_model_parameters[client_id] for client_id in
                                           server.client_ids]
                print('server:' + str(server.server_id) + ' -> ' + 'clients:' + str(server.client_ids))

                # Aggregate client models
                server.train(client_model_parameters)
            else:
                # Internal nodes: Aggregate models from child servers
                child_server_model_parameters = [server_hierarchy[depth_level + 1][child_server].model.state_dict() for
                                                 child_server in server.child_server_ids]

                # Aggregate child server models
                server.train(child_server_model_parameters)

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
            server_parameters = [server.model.state_dict(), parent_server.model.state_dict()]
            # Update edge server models
            server.train(server_parameters)

            # Evaluate the edge server model
            loss, accuracy = server.evaluate(server_test_set)
            loss_and_accuracy_at_level.append((loss, accuracy))

        server_loss_and_accuracy.append(loss_and_accuracy_at_level)

    return server_loss_and_accuracy


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


def change_aggregation_strategy(server_hierarchy: List[Any], drift_recovery_method: str, drift: Drift,
                                clients: List[Client]) -> None:
    """
    Change the aggregation strategy of the leaf servers (only) of the hierarchy.
    :param server_hierarchy: List of servers in the hierarchy
    :param drift_recovery_method: Drift recovery method
    :param drift: Drift instance
    :param clients: List of all client instances
    :return: None
    """
    if drift_recovery_method == constants.RecoveryAlgorithm.FEDAU:
        for server in server_hierarchy[-1]:
            # change the strategy only in the servers where drifted clients are connected
            if server.client_ids in drift.drifted_client_indices:
                server.strategy = strategy.FedAU.aggregator_fn()
    else:
        # if the drift is ended, change the strategy back to FedAvg
        for server in server_hierarchy[-1]:
            server.strategy = strategy.FedAvg.aggregator_fn()
