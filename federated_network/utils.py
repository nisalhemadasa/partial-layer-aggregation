"""
Description: This script contains utility functions required for operations of the federated network.

Author: Nisal Hemadasa
Date: 09-12-2024
Version: 1.0
"""
import copy
from typing import List, OrderedDict, Dict, Any

import constants
from drift_concepts.drift import Drift, apply_drift
from federated_network.client import set_parameters, Client, change_client_drift_recovery_method
from federated_network.server import Server, change_server_aggregation_strategy


def equal_distribution(num_clients: int, num_servers: int) -> List[int]:
    """
    Distribute clients as evenly as possible across servers.
    :param num_clients: Number of clients.
    :param num_servers: Number of servers.
    :return: List of integers representing the number of clients assigned to each server.
    """
    base_clients = num_clients // num_servers
    extra_clients = num_clients % num_servers

    # Distribute extra clients to the first few servers
    return [base_clients + (1 if i < extra_clients else 0) for i in range(num_servers)]


def link_server_hierarchy(server_hierarchy: List[List[Server]]) -> None:
    """
    Link the servers in the hierarchy using a flexible-binary tree structure.
    :param server_hierarchy: List of servers in the hierarchy, where each list represents a level.
    :return: None
    """
    for depth_level in range(len(server_hierarchy) - 1, 0, -1):  # Start from the second-last level
        child_servers = server_hierarchy[depth_level]
        parent_servers = server_hierarchy[depth_level - 1]

        # Divide child servers evenly among parent servers
        num_parents = len(parent_servers)
        num_children = len(child_servers)
        children_per_parent = num_children // num_parents
        extra_children = num_children % num_parents  # Distribute extra children

        child_index = 0  # Track current child server index
        for i, parent_server in enumerate(parent_servers):
            # Assign children to the current parent
            assigned_children = children_per_parent + (1 if i < extra_children else 0)

            for _ in range(assigned_children):
                child_server = child_servers[child_index]
                child_server.parent_server_id = parent_server.server_id
                parent_server.child_server_ids.append(child_server.server_id)
                child_index += 1


def link_clients_to_servers(leaf_servers: List[Server], clients: List[List[Client]]) -> None:
    """
    Determines how the distribution of the clients to the servers at the leaves of the hierarchy should be done. Then
    links the servers to the clients accordingly.
    :param leaf_servers: List of servers at the leaves of the hierarchy
    :param clients: List of client instances
    :return: None
    """
    # Distribute the clients to the servers according to a given ratio (e.g., equally, etc.)
    num_servers = len(leaf_servers)

    for _clients in filter(len, clients):  # Skip empty client lists
        # Get the distribution based on the strategy
        client_distribution = equal_distribution(len(_clients), num_servers)

        if sum(client_distribution) != len(_clients):
            raise ValueError("The distribution strategy must allocate all clients.")

        # Number of clients already linked to the servers
        linked_client_count = 0

        # Distribute clients to servers in a sequence of ascending order of client IDs
        for i, server in enumerate(leaf_servers):
            server.client_ids.extend(
                _clients[j].client_id for j in
                range(linked_client_count, linked_client_count + client_distribution[i]))

            # Assign the server ID to the respective clients to which they are connected to
            # Build a quick-access map from client_id to client instance
            client_map = {client.client_id: client for client in _clients}

            # Assign parent server ID using direct lookup
            for _id in server.client_ids:
                client = client_map.get(_id)
                if client and client.parent_server_id is None:
                    client.parent_server_id = server.server_id

            linked_client_count += client_distribution[i]


def train_client_models(all_clients, sampled_client_ids, servers: List[Server], drift: Drift,
                        simulation_parameters: Dict, drift_recovery_method: str, verbose: bool = False) -> List:
    """
    Train the client models in the network while applying drift if necessary.
    :param all_clients: List of all client instances
    :param sampled_client_ids: List of sampled client IDs
    :param servers: List of Server instance at a given depth level
    :param drift: Drift instance
    :param simulation_parameters: Parameters specifying the simulation scenarios
    :param drift_recovery_method: Drift recovery method to be used by the clients
    :param verbose: Flag to enable verbose logging
    :return: List of loss and accuracy of each client after training
    """
    round_client_loss_and_accuracy = []
    is_server_adaptability = simulation_parameters['is_server_adaptability']

    if verbose:
        print("Training client models...")

    # Apply drift to the clients
    if drift.is_drift:
        # Sample data from the drift applied datasets
        _ = apply_drift(all_clients, drift)

        # TODO: the following needs to add the sampling refresh after drift
        for client in all_clients:
            client.sample_data()  # refresh loaders from drifted datasets
    else:
        for client in all_clients:
            # Sample data from the original datasets
            client.sample_data()

    for client in all_clients:
        # Get the server to which the client is connected
        server = servers[client.parent_server_id]

        if client.client_id in sampled_client_ids:
            if not server.strategy.strategy_name == constants.RecoveryAlgorithm.FEDEX:
                # Download the server model parameters to the client
                set_parameters(client.model, server.model.state_dict())

            if verbose:
                print('server:' + str(server.server_id) + ' -> ' + 'client:' + str(client.client_id))

            if is_server_adaptability:
                # Evaluates the adaptability of the server model to the data
                round_client_loss_and_accuracy.append(client.evaluate())

            # If the client is sampled in this global training round, then train using the server aggregated parameters
            client.fit(drift.is_drift, drift.is_drift_end, server.model.state_dict(), client.client_id,
                       client.drift_recovery_method, drift.drifted_client_indices)
        else:
            # If the client is not sampled, perform local training without server parameters
            client.fit(drift.is_drift, drift.is_drift_end, None, client.client_id, client.drift_recovery_method,
                       drift.drifted_client_indices)

            if is_server_adaptability:
                round_client_loss_and_accuracy.append(client.evaluate())

        if not is_server_adaptability:
            # Evaluate the adaptability of the client models to the data
            round_client_loss_and_accuracy.append(client.evaluate())

    return round_client_loss_and_accuracy


def update_progress(_round, num_training_rounds, verbose=True) -> None:
    """
    Update the progress of the simulation
    :param _round: Current simulation iteration number
    :param num_training_rounds: Total number of training rounds
    :param verbose: Flag to enable verbose logging
    :return: None
    """
    progress = (_round / num_training_rounds) * 100
    if verbose:
        print(f"\rSimulation Percentage completed: {progress:.2f}%", end="")


def handle_drift_for_round(round_idx: int, drift: Drift, server_hierarchy: List[Any],
                           drift_recovery_parameters: Dict, clients: List[Client]) -> None:
    """
    Update drift state for the given round and switch server aggregation when needed.
    Mutates `drift.is_drift` and `drift.current_drift_step` and calls `change_aggregation_strategy` when
    entering/exiting drift steps.
    By default, the initial server aggregation strategy is FedAvg.
    :param round_idx: Current training round index
    :param drift: Drift instance
    :param server_hierarchy: List of servers in the hierarchy
    :param drift_recovery_parameters: Parameters specifying the drift recovery strategies
    :param clients: List of client instances
    :return: None
    """
    drift.current_round = round_idx

    # Outside the global drift window
    if round_idx < drift.drift_start_round or round_idx >= drift.drift_end_round:
        if drift.is_drift:  # execute only once: after the drift period ends
            # Change the aggregation strategy back to FedAvg outside the drift window
            change_server_aggregation_strategy(server_hierarchy, constants.RecoveryAlgorithm.FEDAVG, drift)

            if not drift_recovery_parameters['recovery_method'] == constants.RecoveryAlgorithm.FLUID:
                # Change the clients' (all of them) drift recovery method
                change_client_drift_recovery_method(clients, drift_recovery_parameters['base_aggregation_method'],
                                                    drift.drifted_client_indices)
            else:
                # FLUID
                # Change the clients' (only the drift affected clients) drift recovery method
                change_client_drift_recovery_method(clients, constants.RecoveryAlgorithm.FLUID,
                                                    drift.drifted_client_indices)

            drift.is_drift = False

        # Mark the end of the drift (in contrast to the before the drift starts), only once
        if round_idx >= drift.drift_end_round and not drift.is_drift_end:
            drift.is_drift_end = True
        return

    else:
        next_drift_step_round = drift.drift_step_rounds[drift.current_drift_step + 1]
        if round_idx >= next_drift_step_round:
            if not drift.is_drift:  # execute only once: at the beginning of the drift step
                # The server aggregation strategy needs to change for the FedAU's case, at the start of the drift step.
                change_server_aggregation_strategy(server_hierarchy, drift_recovery_parameters['recovery_method'],
                                                   drift)

                # Change the clients' (all of them) drift recovery method
                change_client_drift_recovery_method(clients, drift_recovery_parameters['recovery_method'],
                                                    drift.drifted_client_indices)

                drift.is_drift = True  # Drift occurs in the current step

            drift.current_drift_step += 1  # Move to the next drift step

            # should not happen in LABEL_SWAP_ONCE's case, because apply_drift() is called once in that case
            if not drift.drift_mode == constants.DriftMode.LABEL_SWAP_ONCE:
                drift.is_already_applied = False  # Reset the flag to apply drift again in the next step
