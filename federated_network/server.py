"""
Description: This module defines a server of the federated network.

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
import copy
from typing import List, OrderedDict, Tuple, Dict, Any

import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader

import constants
import strategy
from distance_metrics.distance_metrics import compute_euclidean_distance_weights
from drift_concepts.drift import Drift
from federated_network.client import DEVICE, Client
from models import ResNet18TinyImageNet
from models.CNNCIFAR100.model import ResNet18CIFAR100, ShallowResNetCIFAR100
from models.utils import SimpleModel, CNNModel, test, split_to_extractor_and_classifier, set_parameters, \
    CNNCIFAR10, CNNCIFAR100, set_parameters_ema, TabularAdultModel, ConvNeXtTinyImageNet
from strategy.FedRC.fedrc import get_fedrc_client_model_params


class Server:
    def __init__(self, _server_id, _abs_id, _strategy, _model, _cluster_count, _fedex_alpha, _client_ids=None):
        self.server_id = _server_id
        self.abs_id = _abs_id  # Absolute ID that keeps a running count of the servers in the server hierarchy
        self.strategy = _strategy
        self.model = _model
        self.fedex_alpha = _fedex_alpha  # EMA weight (alpha) parameter for the FedEx algorithm
        self.client_ids = []  # List of client IDs the server is connected to in the federated network
        self.child_server_ids = []  # List of child server IDs in the server hierarchy
        self.parent_server_id = None  # Parent server ID in the server hierarchy
        self.drift_id = None  # For clustering-based methods, the drift pattern ID assigned to this server (e.g., Oracle)

        # Create a list of models of size '_cluster_count'
        if self.strategy.strategy_name == constants.RecoveryAlgorithm.FEDRC:
            self.multi_models = [copy.deepcopy(_model) for _ in range(_cluster_count)]
            self.model = None
        else:
            self.multi_models = None

    def train(self, client_model_parameters: Dict[str, OrderedDict],
              aux_classifier_parameters: Dict[str, OrderedDict] = None, ema_weight: float = None) -> None:
        """
        Train the server model using the client model parameters.
        :param client_model_parameters: Dictionary of client model parameters
        :param aux_classifier_parameters: Dictionary of auxiliary classifier parameters from drifted clients
        :param ema_weight: EMA weight (alpha) parameter for the FedAU algorithm
        :return: None
        """
        if (self.strategy.strategy_name == constants.RecoveryAlgorithm.FEDAU or
                self.strategy.strategy_name == constants.RecoveryAlgorithm.FLUID):  # TODO: can be changed to strategy name
            if aux_classifier_parameters:
                # check if all the elements in the aux_classifier_parameters are None
                if all(param is None for param in aux_classifier_parameters.values()):
                    return

                # FedAU: For servers with clients with auxiliary classifiers (i.e., leaf servers, having drifted nodes)
                self.strategy.aggregate_models(self.model, client_model_parameters, aux_classifier_parameters,
                                               ema_weight)
        # TODO: remove is_drift to use from the beginning
        elif self.strategy.strategy_name == constants.RecoveryAlgorithm.FEDEX:
            self.strategy.aggregate_models(self.model, client_model_parameters)
        elif self.strategy.strategy_name == constants.RecoveryAlgorithm.FEDRC:
            self.strategy.aggregate_models(self.multi_models, client_model_parameters)
        elif self.strategy.strategy_name == constants.RecoveryAlgorithm.ORACLE:
            self.strategy.aggregate_models(self.model, client_model_parameters)
        else:
            # FedAvg: For internal servers or when there are no drifted clients
            self.strategy.aggregate_models(self.model, client_model_parameters)

    def model_evaluate(self, _test_set: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the server model using the validation data.
        :param _test_set: test data
        :return: loss and accuracy
        """
        loss, accuracy = test(self.model, _test_set)
        return float(loss), float(accuracy)

    def evaluate_multi_models(self, _test_set: DataLoader) -> Tuple[List[float], List[float]]:
        """
        Evaluate all multi-global models in clustering approach (e.g., FedRC, Oracle) in the client on the validation
        data and return the loss and accuracy
        :param _test_set: test data
        :return: list of losses and accuracies. e.g., ([loss1, loss2,...], [acc1, acc2,...])
        """
        losses = []
        accuracies = []
        for model in self.multi_models:
            loss, accuracy = test(model, _test_set)
            losses.append(float(loss))
            accuracies.append(float(accuracy))
        return losses, accuracies

    def average_client_evaluation_results(self, sampled_clients: List[Client]) -> tuple:
        """
        Get the average evaluation performance (accuracy and loss) of the connected clients.
        :param sampled_clients: List of all clients
        :return: average loss and accuracy of the connected clients to this server
        """
        round_server_loss_and_accuracy = []

        # Get the clients connected to this server
        connected_clients = [client for client in sampled_clients if client.client_id in self.client_ids]

        for client in connected_clients:
            round_server_loss_and_accuracy.append(client.evaluate())

        # Position-wise average server loss and accuracy. i.e., take the average loss of all clients (likewise in accuracy)
        arr = np.array(round_server_loss_and_accuracy)  # shape (N, 2)
        return tuple(arr.mean(axis=0))


def model_aggregation_fedrc(server: Server, sampled_clients: List[Client], verbose=False) -> None:
    """
    Aggregate the models of the clients to the server model for FedRC algorithm.
    :param server: Server instance
    :param sampled_clients: List of sampled clients
    :param verbose: Whether to print detailed logs or not
    """
    client_model_parameters = {
        client_id: get_fedrc_client_model_params(sampled_clients[client_id].fedrc_models)
        for client_id in server.client_ids}

    if verbose:
        print('aggregate models: server:' + str(server.server_id) + ' -> ' + 'clients:' + str(server.client_ids))

    # Aggregate client models
    server.train(client_model_parameters, None, None)


def model_aggregation_oracle(server: Server, sampled_clients: List[Client], verbose=False) -> None:
    """
    Aggregate the models of the clients to the server model for Oracle algorithm.
    :param server: Server instance
    :param sampled_clients: List of sampled clients
    :param verbose: Whether to print detailed logs or not
    """
    client_model_parameters = {}

    if sampled_clients:  # Star topology
        # Get model parameters from all participating clients
        client_model_parameters = {client_id: sampled_clients[client_id].model.state_dict()
                                   for client_id in server.client_ids}

    if verbose:
        print('aggregate models: server:' + str(server.server_id) + ' -> ' + 'clients:' + str(server.client_ids))

    # Aggregate client models
    if client_model_parameters:  # check to avoid empty dict error
        server.train(client_model_parameters, None, None)


def model_aggregation_fedau_fluid(server: Server, sampled_clients: List[Client], drift: Drift, ema_weight,
                                  verbose=False) -> None:
    """
    Aggregate the models of the clients to the server model for FedAU and FLUID algorithm.
    :param server: Server instance
    :param sampled_clients: List of sampled clients
    :param drift: Drift instance
    :param ema_weight: EMA weight (alpha) parameter for the FedAU algorithm
    :param verbose: Whether to print detailed logs or not
    """
    # Get auxiliary classifier parameters from drifted clients only
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

        # Get model parameters from all participating clients
        client_model_parameters = {client_id: sampled_clients[client_id].model.state_dict()
                                   for client_id in server.client_ids}

        if verbose:
            print('aggregate models: server:' + str(server.server_id) + ' -> ' + 'clients:' + str(server.client_ids))

        server.train(client_model_parameters, client_aux_classifier_parameters, ema_weight)


def model_aggregation_fedavg(server: Server, sampled_clients: List[Client], server_hierarchy, depth_level,
                             verbose=False) -> None:
    """
    In star-topology: Aggregate the models of the clients to the server model for FedAvg, RRT, FedEx algorithm.
    In hierarchical topology: Aggregate the models of the clients and child servers to the parent server model for
    FedAvg, RRT,
    :param server: Server instance
    :param sampled_clients: List of sampled clients
    :param server_hierarchy: List of servers in the hierarchy
    :param depth_level: Current depth level in the server hierarchy
    :param verbose: Whether to print detailed logs or not
    :return: None
    """
    if sampled_clients:  # Star topology
        # Get model parameters from all participating clients
        model_parameters = {client_id: sampled_clients[client_id].model.state_dict()
                            for client_id in server.client_ids}

        if verbose:
            print('aggregate client models: server:' + str(server.server_id) + ' -> ' + 'clients:' + str(
                server.client_ids))

    elif server_hierarchy and depth_level is not None:  # Hierarchical topology
        # Get model parameters from all child servers
        model_parameters = {
            child_server: server_hierarchy[depth_level + 1][child_server].model.state_dict()
            for child_server in server.child_server_ids}

        if verbose:
            print('aggregate server models.')

    else:
        raise ValueError(
            "model_aggregation_fedavg: Either sampled_clients or server_hierarchy and depth_level must be provided")

    # Aggregate client models
    server.train(model_parameters, None, None)


def model_aggregation(server_hierarchy: List[List[Server]], server_test_set: DataLoader, sampled_clients: List[Client],
                      drift: Drift, ema_weight: float, _is_server_has_test_data: bool,
                      verbose=False) -> List:
    """
    Aggregate the models of the clients to the server model.
    :param server_hierarchy: List of servers in the hierarchy
    :param server_test_set: List of test data for server model evaluation, once the aggregation is done
    :param sampled_clients: List of sampled clients
    :param drift: Drift instance
    :param ema_weight: EMA weight (alpha) parameter for the FedAU algorithm
    :param _is_server_has_test_data: Boolean flag to indicate whether server possesses test data to do in-server
    evaluations after client model aggregation. Else, the average of client evaluation results will be used as server
    evaluation performance.
    :param verbose: Whether to print detailed logs or not
    :return: List of loss and accuracy at each level of the server hierarchy; outer list: server hierarchy levels,
    inner list: loss and accuracy Tuple at each level (loss, accuracy)
    """
    # Store the loss and accuracy at each level of the server model hierarchy
    server_loss_and_accuracy = []

    # Evaluate server model on the upward traversal aggregation  only if the hierarchy has more than one level. Else,
    # this evaluation will be redundant as it is already done during the server model distribution phase.
    is_evaluate_server_model = False
    if _is_server_has_test_data and len(server_hierarchy) > 1:
        is_evaluate_server_model = True

    # TODO: refactor: aggregation and evaluation could be separated
    # Aggregate the models of the clients to the server model.Start by aggregating the leaves and move up the hierarchy
    for depth_level in range(len(server_hierarchy) - 1, -1, -1):
        loss_and_accuracy_at_level = []

        for server in server_hierarchy[depth_level]:
            if depth_level == len(server_hierarchy) - 1:  # ==== Leaf servers ====
                # Get client (learning) model parameters
                if server.strategy.strategy_name == constants.RecoveryAlgorithm.FEDRC:
                    model_aggregation_fedrc(server, sampled_clients, verbose=verbose)

                elif server.strategy.strategy_name == constants.RecoveryAlgorithm.ORACLE:
                    model_aggregation_oracle(server, sampled_clients, verbose=verbose)

                elif server.strategy.strategy_name in {constants.RecoveryAlgorithm.FEDAU,
                                                       constants.RecoveryAlgorithm.FLUID}:
                    model_aggregation_fedau_fluid(server, sampled_clients, drift, ema_weight, verbose=verbose)

                elif server.strategy.strategy_name in {constants.RecoveryAlgorithm.FEDAVG,
                                                       constants.RecoveryAlgorithm.RRT,
                                                       constants.RecoveryAlgorithm.FEDEX}:
                    model_aggregation_fedavg(server, sampled_clients, None, None, verbose=verbose)
                else:
                    raise ValueError("Server.model_aggregation: Unsupported recovery algorithm name")

            else:  # ==== Upper hierarchical servers (aggregate models from child servers) ====
                model_aggregation_fedavg(server, None, server_hierarchy, depth_level, verbose=verbose)

            # Evaluate the server model (not for FedEx)
            if is_evaluate_server_model and not server.strategy.strategy_name == constants.RecoveryAlgorithm.FEDEX:
                loss, accuracy = server.model_evaluate(server_test_set)
                loss_and_accuracy_at_level.append((loss, accuracy))

        if is_evaluate_server_model and not server.strategy.strategy_name == constants.RecoveryAlgorithm.FEDEX:
            server_loss_and_accuracy.append(loss_and_accuracy_at_level)

    if is_evaluate_server_model and not server.strategy.strategy_name == constants.RecoveryAlgorithm.FEDEX:
        server_loss_and_accuracy.reverse()  # Reverse the list to get the root first, to be consistent throughout the code

    return server_loss_and_accuracy


def model_distribution_hierarchy(server_hierarchy: List[List[Server]]) -> None:
    """
    The aggregated models are distributed down the hierarchy. I.e., the edge models are updated by the global model and
    leaf-server models are updated by the edge models.
    :param server_hierarchy: List of servers in the hierarchy
    :return: None
    """
    # Aggregate the global server to the edge server down the hierarchy starting from the leaf nodes
    for depth_level in range(len(server_hierarchy) - 1):
        # Update the edge server model with the global server model
        for server in server_hierarchy[depth_level + 1]:
            # Get global server parameters and update the edge server model
            parent_server = server_hierarchy[depth_level][server.parent_server_id]
            server_parameters = {server.server_id: server.model.state_dict(),
                                 parent_server.server_id: parent_server.model.state_dict()}
            # Update edge server models
            server.train(server_parameters, None, None)


def model_distribution_fedex(servers: List[Server], all_clients: List[Client]) -> None:
    """
    Distribute the server model to (1)drifted clients (2)all clients.
    TODO: implementation has to be expanded in the case of a hierarchical server structure
    :param servers: The aggregated server models (to be distributed)
    :param all_clients: List of all clients
    """
    for client in all_clients:  # TODO: here fedex is done to all cleints and not only drifted clients
        # Get the server to which the client is connected
        server = servers[client.parent_server_id]
        # get the extractor of the server model
        server_extractor, _ = split_to_extractor_and_classifier(server.model, None, server.model.get_model_type())

        if server.fedex_alpha:  # if FedEx is used with EMA
            set_parameters_ema(client.model, server_extractor, server.fedex_alpha, False)
        else:  # Load the composite unlearning model ({E, W_hat}) to the server model
            set_parameters(client.model, server_extractor, False)


def model_distribution_fedrc(server_list: List[Server], sampled_clients: List[Client]) -> None:
    """
    The aggregated server models are distributed to each participating clients. Each client get all k server models.
    I.e., each client also has k models in it.
    :param server_list: List of servers in the hierarchy
    :param sampled_clients: List of sampled clients
    :return: None
    """
    server = server_list[0]  # TODO: implemented only for single server topology
    for client_idx, client in enumerate(sampled_clients):
        for cluster_idx in range(len(server.multi_models)):
            set_parameters(client.fedrc_models[cluster_idx], server.multi_models[cluster_idx].state_dict())


def server_hierarchy_evaluate(server_hierarchy: List[Server], server_test_set: DataLoader,
                              all_clients: List[Client],
                              _is_server_has_test_data: bool, _drift_recovery_method: str) -> List:
    """
    The aggregated models are distributed to the client models.
    :param server_hierarchy: List of servers
    :param server_test_set: List of test data for server model evaluation, once the aggregation is done
    :param all_clients: List of all clients
    :param _is_server_has_test_data: Boolean flag to indicate whether server possesses test data to do in-server
    evaluations after client model aggregation. Else, the average of client evaluation results will be used as server
    evaluation performance.
    :param _drift_recovery_method: Drift recovery method
    :return: None
    """
    # Store the loss and accuracy at each level of the server model hierarchy
    server_loss_and_accuracy = []
    global_avg_loss_and_accuracy = None

    # Evaluate the accuracy of the root server model
    global_server = server_hierarchy[0][0]

    # TODO: this part has to be refactored
    if _drift_recovery_method == constants.RecoveryAlgorithm.FEDRC:
        # FEDRC: Evaluate all multiple models in the servers
        losses, accuracies = global_server.evaluate_multi_models(
            server_test_set)  # returns ([loss1, loss2,...], [acc1, acc2,...])
        server_loss_and_accuracy.append((losses, accuracies))
    elif _drift_recovery_method == constants.RecoveryAlgorithm.ORACLE:
        # ORACLE: multiple-server, clustering-based
        if _is_server_has_test_data:
            for server in server_hierarchy[0]:
                if server.client_ids:  # to avoid empty server case
                    loss, accuracy = server.model_evaluate(server_test_set)
                    server_loss_and_accuracy.append([(loss, accuracy)])
                else:  # TODO: give a better solution for empty server case
                    server_loss_and_accuracy.append([(0.0, 0.0)])  # placeholder for empty server
        else:
            for server in server_hierarchy[0]:
                if server.client_ids:  # to avoid empty server case
                    loss, accuracy = server.average_client_evaluation_results(all_clients)
                    server_loss_and_accuracy.append([(loss, accuracy)])
                else:  # TODO: give a better solution for empty server case
                    server_loss_and_accuracy.append([(0.0, 0.0)])  # placeholder for empty server
    else:
        if _is_server_has_test_data:
            loss, accuracy = global_server.model_evaluate(server_test_set)
        else:
            loss, accuracy = global_server.average_client_evaluation_results(all_clients)

        # Store the loss and accuracy of the global server model
        server_loss_and_accuracy.append([(loss, accuracy)])

    # Evaluate the edge servers down the hierarchy starting upto the leaf servers
    for depth_level in range(len(server_hierarchy) - 1):
        loss_and_accuracy_at_level = []

        # Update the edge server model with the global server model
        for server in server_hierarchy[depth_level + 1]:
            # Evaluate the edge server model
            if _is_server_has_test_data:
                loss, accuracy = server.model_evaluate(server_test_set)
            else:
                loss, accuracy = server.average_client_evaluation_results(all_clients)

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
    if (drift_recovery_method == constants.RecoveryAlgorithm.FEDAU or
            drift_recovery_method == constants.RecoveryAlgorithm.FLUID):
        for server in server_hierarchy[-1]:  # applied only to leaf servers
            # change the strategy only in the servers where drifted clients are connected
            # drifted = set(drift.drifted_client_indices or [])  # makes sure it's at least an empty set and not None
            # if set(server.client_ids) & drifted:  # checks if there is any intersection
            server.strategy = strategy.FedAU.aggregator_fn()

    elif drift_recovery_method == constants.RecoveryAlgorithm.FEDEX:
        # TODO: implement for (1) drifted clients + during drift, (2) all clients + during drift (3) all clients +all times
        for server in server_hierarchy[-1]:  # applied only to leaf servers
            # # change the strategy only in the servers where drifted clients are connected
            # drifted = set(drift.drifted_client_indices or [])  # makes sure it's at least an empty set and not None
            # if set(server.client_ids) & drifted:  # checks if there is any intersection
            server.strategy = strategy.FedEx.aggregator_fn()

    elif drift_recovery_method == constants.RecoveryAlgorithm.ORACLE:
        for idx, server in enumerate(server_hierarchy[-1]):  # applied only to leaf servers
            # assign drift_pattern ID for each server, so the clients with same drift pattern are connected to it.
            server.drift_id = drift.unique_drift_ids[idx]
            server.strategy = strategy.Oracle.aggregator_fn()

    else:
        # if the drift is ended, change the strategy back to FedAvg
        for server in server_hierarchy[-1]:
            server.strategy = strategy.FedAvg.aggregator_fn()


def server_fn(server_id: int, dataset_name: str, server_abs_id: int, drift_recovery_method: str, cluster_count: int,
              fedex_alpha: float, ) -> Server:
    """
    Create a server instances on demand for the optimal use of resources.
    :param server_id: Server ID
    :param dataset_name: Name of the dataset
    :param server_abs_id: Absolute server ID; a running count of all the servers created
    :param drift_recovery_method: Drift recovery method to be used by the client
    :param cluster_count: number of models (clusters) in the server (for multi-global-model methods, e.g.FedRC, Oracle)
    :param fedex_alpha: EMA weight (alpha) parameter for the FedEx algorithm
    :returns Server: A Server instance.
    """
    if drift_recovery_method == constants.RecoveryAlgorithm.FEDRC:
        aggregator_strategy = strategy.FedRC.aggregator_fn()
    elif drift_recovery_method == constants.RecoveryAlgorithm.ORACLE:
        aggregator_strategy = strategy.Oracle.aggregator_fn()
    elif drift_recovery_method == constants.RecoveryAlgorithm.FEDEX:  # TODO: remove after testing
        aggregator_strategy = strategy.FedEx.aggregator_fn()  # TODO: remove after testing
    else:
        aggregator_strategy = strategy.FedAvg.aggregator_fn()

    # model = SimpleModel().to(DEVICE)
    if dataset_name == constants.DatasetNames.MNIST or dataset_name == constants.DatasetNames.F_MNIST:
        model = CNNModel().to(DEVICE)
    elif dataset_name == constants.DatasetNames.CIFAR_10:
        model = CNNCIFAR10().to(DEVICE)
    elif dataset_name == constants.DatasetNames.CIFAR_100:
        # model = ShallowResNetCIFAR100().to(DEVICE)
        model = ResNet18CIFAR100().to(DEVICE)
        # model = CNNCIFAR100().to(DEVICE)
    elif dataset_name == constants.DatasetNames.TINY_IMAGENET_200:
        model = ConvNeXtTinyImageNet().to(DEVICE)
    elif dataset_name == constants.DatasetNames.ADULT:
        model = TabularAdultModel().to(DEVICE)
    else:
        raise ValueError("Unsupported dataset name")

    return Server(_server_id=server_id, _abs_id=server_abs_id, _strategy=aggregator_strategy, _model=model,
                  _cluster_count=cluster_count, _fedex_alpha=fedex_alpha)
