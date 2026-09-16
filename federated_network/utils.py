"""
Description: This script contains utility functions required for operations of the federated network.

Author: Nisal Hemadasa
Date: 09-12-2024
Version: 1.0
"""
import copy
from typing import List, OrderedDict, Dict, Any

import constants
from drift_concepts.drift import apply_drift, Drift
from federated_network.client import set_parameters, Client, change_client_drift_recovery_method, set_client_drift_ids
from federated_network.server import Server, change_server_aggregation_strategy
from strategy.FedRC import fedrc


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


def first_server_lead_distribution(num_clients: int, num_servers: int) -> List[int]:
    """
    The 0th indexed (first server) get all the clients, others get none.
    :param num_clients: Number of clients.
    :param num_servers: Number of servers.
    :return: List of integers representing the number of clients assigned to each server.
    """
    return [num_clients] + [0] * (num_servers - 1)


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

    for _clients in list(filter(len, clients)):  # Skip empty client lists
        # Get the distribution based on the strategy
        # client_distribution = equal_distribution(len(_clients), num_servers)
        client_distribution = first_server_lead_distribution(len(_clients), num_servers)

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


def apply_drift_to_clients(drift: Drift, all_clients: list[Client]) -> None:
    """
    #TODO: this function could be moved to drift_concepts/utils.py
    #TODO: to do that, a proper planning of package imports needs to be done
    Apply drift to the selected clients in the network if necessary.
    :param drift: Drift instance
    :param all_clients: List of all client instances
    :return: None
    """
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


def get_drifted_classes(drift: Drift) -> set[int]:
    """
    Return labels referenced by the configured label-drift patterns.
    :param drift: Drift instance containing pattern specifications.
    :return: Set of affected labels; empty for non-label drift configurations.
    """
    if drift.drift_mode not in {constants.DriftMode.LABEL_SWAP_ONCE,
                                constants.DriftMode.LABEL_SWAP_INCREMENTAL_STEPS}:
        return set()
    if not drift.drift_pattern_id_map:
        return set()
    return {label for class_pairs in drift.drift_pattern_id_map.values()
            for class_pair in class_pairs for label in class_pair}


def evaluate_clients_for_stage(all_clients: List[Client], servers: List[Server], round_idx: int,
                               sampled_client_ids: List[int], stage: str, drift: Drift,
                               include_drifted_classes: bool = False) -> Dict:
    """
    Evaluate local or assigned-server models without changing persistent client model state.
    :param all_clients: All clients, returned in stable client-ID order.
    :param servers: Leaf servers used to resolve each client's assigned server model.
    :param round_idx: Current aggregation round.
    :param sampled_client_ids: IDs associated with this round's aggregation event.
    :param stage: local_before_download, global_after_download, or local_after_training.
    :param drift: Drift metadata used for optional class-subset evaluation.
    :param include_drifted_classes: Whether to include drifted-class metrics.
    :return: Structured evaluation record.
    """
    valid_stages = {'local_before_download', 'global_after_download', 'local_after_training'}
    if stage not in valid_stages:
        raise ValueError(f"Unsupported client evaluation stage: {stage}")
    server_map = {server.server_id: server for server in servers}
    sampled_ids = set(sampled_client_ids)
    target_classes = get_drifted_classes(drift) if include_drifted_classes else set()
    records = []

    for client in sorted(all_clients, key=lambda item: item.client_id):
        server = server_map[client.parent_server_id]
        if client.fedrc_models is not None:
            model_pairs = []
            for model_id, local_model in enumerate(client.fedrc_models):
                model = server.multi_models[model_id] if stage == 'global_after_download' else local_model
                model_pairs.append((model_id, model))
        else:
            model = server.model if stage == 'global_after_download' else client.model
            model_pairs = [('primary', model)]

        for model_id, model in model_pairs:
            loss, accuracy = client.evaluate(model)
            record = {
                'client_id': client.client_id,
                'parent_server_id': server.server_id,
                'server_abs_id': server.abs_id,
                'model_id': model_id,
                'model_role': 'assigned_server' if stage == 'global_after_download' else 'local_client',
                'participated': client.client_id in sampled_ids,
                'loss': loss,
                'accuracy': accuracy
            }
            if include_drifted_classes:
                class_loss, class_accuracy, sample_count = client.evaluate_drifted_classes(target_classes, model)
                record['drifted_class_metrics'] = {
                    'classes': sorted(target_classes),
                    'loss': class_loss,
                    'accuracy': class_accuracy,
                    'sample_count': sample_count
                }
            records.append(record)

    return {'round': round_idx, 'stage': stage, 'clients': records}


def evaluate_ditto_personalized_clients(all_clients: List[Client], servers: List[Server], round_idx: int,
                                        sampled_client_ids: List[int], drift: Drift,
                                        include_drifted_classes: bool = False) -> Dict:
    """
    Evaluate initialized Ditto personalized models after local training.
    :param all_clients: All clients, returned in stable client-ID order.
    :param servers: Leaf servers used to record parent-server identity.
    :param round_idx: Current local-training round.
    :param sampled_client_ids: IDs participating in the local-training event.
    :param drift: Drift metadata used for optional class-subset evaluation.
    :param include_drifted_classes: Whether to include affected-class metrics.
    :return: Structured personalized evaluation record.
    """
    server_map = {server.server_id: server for server in servers}
    sampled_ids = set(sampled_client_ids)
    target_classes = get_drifted_classes(drift) if include_drifted_classes else set()
    records = []

    for client in sorted(all_clients, key=lambda item: item.client_id):
        server = server_map[client.parent_server_id]
        loss, accuracy = client.evaluate_ditto_personalized()
        record = {
            'client_id': client.client_id,
            'parent_server_id': server.server_id,
            'server_abs_id': server.abs_id,
            'model_id': 'personalized',
            'model_role': 'ditto_personalized',
            'participated': client.client_id in sampled_ids,
            'loss': loss,
            'accuracy': accuracy
        }
        if include_drifted_classes:
            class_loss, class_accuracy, sample_count = \
                client.evaluate_ditto_personalized_drifted_classes(target_classes)
            record['drifted_class_metrics'] = {
                'classes': sorted(target_classes),
                'loss': class_loss,
                'accuracy': class_accuracy,
                'sample_count': sample_count
            }
        records.append(record)

    return {'round': round_idx, 'stage': 'local_after_training', 'clients': records}


def build_ditto_state_log(clients: List[Client], drift_recovery_parameters: Dict) -> Dict:
    """
    Build serializable Ditto configuration and client-state metadata.
    :param clients: Ditto client instances.
    :param drift_recovery_parameters: Resolved experiment parameters.
    :return: Versioned Ditto state record.
    """
    parameter_names = [
        'ditto_lambda', 'ditto_learning_rate', 'ditto_personal_epochs', 'ditto_eval_personalized',
        'ditto_dynamic_lambda', 'ditto_lambda_candidates', 'ditto_validation_fraction',
        'ditto_validation_seed'
    ]
    return {
        'schema_version': 1,
        'strategy': constants.RecoveryAlgorithm.DITTO,
        'upload_model_role': 'local_client',
        'personalized_model_role': 'ditto_personalized',
        'personal_optimizer_state_policy': 'recreated_per_round',
        'parameters': {name: copy.deepcopy(drift_recovery_parameters.get(name)) for name in parameter_names},
        'clients': [
            {
                'client_id': client.client_id,
                'parent_server_id': client.parent_server_id,
                'ditto_initialized': bool(client.ditto_initialized),
                'has_personalized_model': client.ditto_personal_model is not None,
                'validation_sample_count': (len(client.ditto_validation_indices)
                                            if client.ditto_validation_indices is not None else 0),
                'training_pool_sample_count': (len(client.ditto_training_indices)
                                               if client.ditto_training_indices is not None else None),
                'selected_lambda': client.ditto_selected_lambda
            }
            for client in sorted(clients, key=lambda item: item.client_id)
        ]
    }


def build_ditto_lambda_record(clients: List[Client], round_idx: int) -> Dict:
    """
    Build one round of dynamic Ditto lambda decisions.
    :param clients: Ditto client instances.
    :param round_idx: Current local-training round.
    :return: Structured per-client lambda decision record.
    """
    return {
        'round': round_idx,
        'stage': 'local_after_training',
        'clients': [
            {
                'client_id': client.client_id,
                'parent_server_id': client.parent_server_id,
                'selected_lambda': client.ditto_selected_lambda,
                'decisions': copy.deepcopy(client.ditto_lambda_decisions)
            }
            for client in sorted(clients, key=lambda item: item.client_id)
        ]
    }


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

    for client in all_clients:
        # Get the server to which the client is connected
        server = servers[client.parent_server_id]

        if client.client_id in sampled_client_ids:
            # If the client is sampled in this global training round, then execute the following

            if verbose:
                print('server:' + str(server.server_id) + ' -> ' + 'client:' + str(client.client_id))

            if is_server_adaptability:
                # Evaluates the adaptability of the server model to the data
                round_client_loss_and_accuracy.append(client.evaluate())

            # Download the server model parameters to the client.
            if not server.strategy.strategy_name in [constants.RecoveryAlgorithm.FEDEX,
                                                     constants.RecoveryAlgorithm.FEDRC,
                                                     constants.RecoveryAlgorithm.FEDAU]:
                # FedRC and FedEx strategies have already done this in network.model_distribution_fedex and
                # network.model_distribution_fedrc functions.
                # FedAU does this inside the client.fit() function.
                set_parameters(client.model, server.model.state_dict())

            # Train the client models using the server aggregated parameters
            if server.strategy.strategy_name == constants.RecoveryAlgorithm.FEDRC:
                client.fit(drift.is_drift, drift.is_drift_end, None, client.client_id,
                           client.drift_recovery_method, drift.drifted_client_indices)
            else:
                client.fit(drift.is_drift, drift.is_drift_end, server.model.state_dict(), client.client_id,
                           client.drift_recovery_method, drift.drifted_client_indices)
        else:
            # If the client is not sampled, perform local training without server parameters
            client.fit(drift.is_drift, drift.is_drift_end, None, client.client_id, client.drift_recovery_method,
                       drift.drifted_client_indices)

            if is_server_adaptability:
                round_client_loss_and_accuracy.append(client.evaluate())

        if not is_server_adaptability:
            if drift_recovery_method == constants.RecoveryAlgorithm.FEDRC:
                losses, accuracies = client.evaluate_fedrc_models()  # returns ([loss1, loss2,...], [acc1, acc2,...])
                round_client_loss_and_accuracy.append((losses, accuracies))
            else:
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


def link_clients_to_servers_by_drift_id(clients: List[Client], server_hierarchy: List[Any]) -> None:
    """
    Clustering clients to servers based on their drift IDs.
    :param clients: List of client instances
    :param server_hierarchy: List of servers in the hierarchy
    :return: None
    """
    for server in server_hierarchy:
        server.client_ids = []  # reset the server's client IDs

        for client in clients:
            if client.drift_id == server.drift_id:
                client.parent_server_id = server.server_id
                server.client_ids.append(client.client_id)


def handle_after_drift_configurations(drift: Drift, server_hierarchy: List[Any], drift_recovery_parameters: Dict,
                                      clients: List[Client]) -> None:
    """
    Set parameter configurations for the operations after the drift ends.
    :param drift: Drift instance
    :param server_hierarchy: List of servers in the hierarchy
    :param drift_recovery_parameters: Parameters specifying the drift recovery strategies
    :param clients: List of client instances
    :return: None
    """
    if drift.is_drift:  # execute only once: after the drift period ends
        if not drift_recovery_parameters['recovery_method'] == constants.RecoveryAlgorithm.FEDRC:
            # Ditto is a run-wide personalized method. Other recovery methods return to the configured base method.
            after_drift_method = drift_recovery_parameters['base_aggregation_method']
            if drift_recovery_parameters['recovery_method'] == constants.RecoveryAlgorithm.DITTO:
                after_drift_method = constants.RecoveryAlgorithm.DITTO
            change_server_aggregation_strategy(server_hierarchy, after_drift_method, drift)

            if not drift_recovery_parameters['recovery_method'] == constants.RecoveryAlgorithm.FLUID:
                if drift_recovery_parameters['recovery_method'] == constants.RecoveryAlgorithm.DITTO:
                    for client in clients:
                        client.drift_recovery_method = constants.RecoveryAlgorithm.DITTO
                else:
                    change_client_drift_recovery_method(clients, after_drift_method,
                                                        drift.drifted_client_indices)
            else:
                # FLUID
                # Change the clients' (only the drift affected clients) drift recovery method
                change_client_drift_recovery_method(clients, constants.RecoveryAlgorithm.FLUID,
                                                    drift.drifted_client_indices)

        drift.is_drift = False


def handle_in_drift_configurations(drift: Drift, server_hierarchy: List[Any], drift_recovery_parameters: Dict,
                                   clients: List[Client]) -> None:
    """
    Set parameter configurations for the operations happening during the drift periods.
    :param drift: Drift instance
    :param server_hierarchy: List of servers in the hierarchy
    :param drift_recovery_parameters: Parameters specifying the drift recovery strategies
    :param clients: List of client instances
    :return: None
    """
    if not drift.is_drift:  # execute only once: at the beginning of the drift step
        # The server aggregation strategy needs to change for the FedAU's case, at the start of the drift step.
        change_server_aggregation_strategy(server_hierarchy, drift_recovery_parameters['recovery_method'],
                                           drift)

        # Change the clients' (all of them) drift recovery method. (This is not needed for Oracle.)
        change_client_drift_recovery_method(clients, drift_recovery_parameters['recovery_method'],
                                            drift.drifted_client_indices)

        drift.is_drift = True  # Drift occurs in the current step

    drift.current_drift_step += 1  # Move to the next drift step

    # For Oracle: Assign each drift affected client a drift group
    if not drift.is_synchronous:
        # set the drifted_client_indices to the current_drift_step's drifted client indices
        drift.drifted_client_indices = drift.drift_clustered_client_indices[drift.current_drift_step]

        set_client_drift_ids(clients, drift.drifted_client_indices, drift.unique_drift_ids,
                             drift.drift_patterns_over_time[drift.current_drift_step])
        if drift_recovery_parameters['recovery_method'] == constants.RecoveryAlgorithm.ORACLE:
            # Cluster clients to servers based on their drift IDs
            link_clients_to_servers_by_drift_id(clients, server_hierarchy[-1])
            p=0

    # should not happen in LABEL_SWAP_ONCE's case, because apply_drift() is called once in that case
    if not drift.drift_mode == constants.DriftMode.LABEL_SWAP_ONCE:
        drift.is_already_applied = False  # Reset the flag to apply drift again in the next step


def handle_drift_for_round(round_idx: int, drift: Drift, server_hierarchy: List[Any],
                           drift_recovery_parameters: Dict, clients: List[Client]) -> None:
    """
    Update drift state for the given round and switch server aggregation when needed.
    Mutates `drift.is_drift` and `drift.current_drift_step` and calls `change_aggregation_strategy` when
    entering/exiting drift steps.
    By default, the initial server aggregation strategy is FedAvg for non-clustering approaches (else Oracle or FedRC).
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
        # parameter updates for the operations after the drift ends
        handle_after_drift_configurations(drift, server_hierarchy, drift_recovery_parameters, clients)

        # Mark the end of the drift (in contrast to the before the drift starts), only once
        if round_idx >= drift.drift_end_round and not drift.is_drift_end:
            drift.is_drift_end = True
        return
    else:  # Inside the global drift window
        next_drift_step_round = drift.drift_step_rounds[drift.current_drift_step + 1]
        if round_idx >= next_drift_step_round:
            # parameter updates for the next drift step
            handle_in_drift_configurations(drift, server_hierarchy, drift_recovery_parameters, clients)
