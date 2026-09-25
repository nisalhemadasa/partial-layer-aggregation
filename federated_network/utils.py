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
from strategy.FedBABU import initialize_fedbabu_personal_models
from strategy.FedRC import fedrc
from strategy.FairFedDrift.utils import (ClientDataHistory, build_fairfeddrift_history_loaders,
                                         evaluate_fairfeddrift_loss, prepare_fairfeddrift_decision_loader)


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


def link_clients_to_fairfeddrift_servers(servers: List[Server], clients: List[Client],
                                         cluster_ids: List[int],
                                         client_assignments: Dict[int, int]) -> Dict[int, int]:
    """
    Link learned FairFedDrift cluster IDs to positional server slots and clients.
    :param servers: Flat server list ordered to match cluster_ids.
    :param clients: All clients participating in the current FairFedDrift round.
    :param cluster_ids: Active learned cluster IDs in server-list order.
    :param client_assignments: Client ID to learned cluster ID mapping.
    :return: Learned cluster ID to server-list position mapping.
    """
    if not servers or len(servers) != len(cluster_ids):
        raise ValueError("FairFedDrift requires one flat server per active cluster.")
    if len(set(cluster_ids)) != len(cluster_ids) or any(
            isinstance(cluster_id, bool) or not isinstance(cluster_id, int) or cluster_id < 0
            for cluster_id in cluster_ids):
        raise ValueError("FairFedDrift cluster IDs must be unique non-negative integers.")
    client_ids = [client.client_id for client in clients]
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("FairFedDrift client IDs must be unique.")
    if set(client_assignments) != set(client_ids):
        raise ValueError("FairFedDrift must provide one learned assignment per client.")

    server_positions = {cluster_id: position for position, cluster_id in enumerate(cluster_ids)}
    if any(cluster_id not in server_positions for cluster_id in client_assignments.values()):
        raise ValueError("FairFedDrift client assignment references an inactive cluster.")
    for server in servers:
        server.client_ids.clear()
    clients_by_id = {client.client_id: client for client in clients}
    for client_id, cluster_id in client_assignments.items():
        position = server_positions[cluster_id]
        servers[position].client_ids.append(client_id)
        # train_client_models() indexes the flat server list with this field.
        clients_by_id[client_id].parent_server_id = position
    return server_positions


def initialize_fairfeddrift_runtime(servers: List[Server]) -> Any:
    """
    Share FairFedDrift state and bind its initial cluster to the live server model.
    :param servers: Flat FairFedDrift server list for this simulation run.
    :return: The single shared FairFedDrift strategy instance.
    """
    if not servers:
        raise ValueError("FairFedDrift runtime requires at least one server.")
    if any(server.strategy.strategy_name != constants.RecoveryAlgorithm.FAIRFEDDRIFT for server in servers):
        raise ValueError("Activate FairFedDrift on every leaf server before initializing its runtime state.")
    shared_strategy = next((server.fairfeddrift_strategy for server in servers
                             if getattr(server, 'fairfeddrift_strategy', None) is not None), None)
    if shared_strategy is None:
        shared_strategy = next((server.strategy for server in servers
                                if server.strategy.strategy_name == constants.RecoveryAlgorithm.FAIRFEDDRIFT), None)
    if shared_strategy is None:
        raise ValueError("No retained FairFedDrift strategy state is available for this run.")
    if not hasattr(shared_strategy, '_next_server_abs_id'):
        shared_strategy._next_server_abs_id = max(server.abs_id for server in servers) + 1
    for server in servers:
        server.fairfeddrift_strategy = shared_strategy
        server.strategy = shared_strategy

    if not shared_strategy.cluster_models:
        initial_cluster_id = shared_strategy.initialize_clusters(servers[0].model)
        servers[0].model = shared_strategy.cluster_models[initial_cluster_id]
        servers[0].fairfeddrift_cluster_id = initial_cluster_id
    elif len(shared_strategy.cluster_models) == 1:
        # Also repair ownership if a caller initialized the registry before linking.
        initial_cluster_id = next(iter(shared_strategy.cluster_models))
        if initial_cluster_id == 0:
            servers[0].model = shared_strategy.cluster_models[initial_cluster_id]
            servers[0].fairfeddrift_cluster_id = initial_cluster_id
    return shared_strategy


def synchronize_fairfeddrift_servers(servers: List[Server], shared_strategy: Any) -> List[int]:
    """
    Make one flat server slot for each active learned cluster, preserving model identity.
    :param servers: Mutable flat server list used by the network hierarchy.
    :param shared_strategy: Shared FairFedDrift state with active cluster models.
    :return: Active cluster IDs in their corresponding server-list order.
    """
    if not servers or not shared_strategy.cluster_models:
        raise ValueError("FairFedDrift needs active clusters and a template server.")
    active_cluster_ids = list(shared_strategy.cluster_models)
    servers_by_cluster = {}
    for position, server in enumerate(servers):
        cluster_id = getattr(server, 'fairfeddrift_cluster_id', position)
        if cluster_id in servers_by_cluster:
            raise ValueError("FairFedDrift has duplicate server mappings for a cluster.")
        servers_by_cluster[cluster_id] = server

    template_server = servers[0]
    next_abs_id = max(shared_strategy._next_server_abs_id,
                      max(server.abs_id for server in servers) + 1)
    ordered_servers = []
    for position, cluster_id in enumerate(active_cluster_ids):
        if cluster_id in servers_by_cluster:
            server = servers_by_cluster[cluster_id]
        else:
            server = copy.deepcopy(template_server)
            server.abs_id = next_abs_id
            next_abs_id += 1
        server.server_id = position
        server.fairfeddrift_cluster_id = cluster_id
        server.fairfeddrift_strategy = shared_strategy
        server.strategy = shared_strategy
        server.model = shared_strategy.cluster_models[cluster_id]
        server.client_ids = []
        server.child_server_ids = []
        server.parent_server_id = None
        server.drift_id = None
        server.multi_models = None
        ordered_servers.append(server)

    servers[:] = ordered_servers
    shared_strategy._next_server_abs_id = next_abs_id
    return active_cluster_ids


def run_fairfeddrift_decisions(servers: List[Server], clients: List[Client],
                              decision_loaders: Dict[int, Any] = None) -> List[Dict]:
    """
    Score every client against one frozen candidate set, then apply all assignments.
    :param servers: Mutable flat server list for the active learned clusters.
    :param clients: All clients with current-round ordinary training loaders.
    :param decision_loaders: Optional prepared client-ID to frozen current-data loader mapping.
    :return: Per-client candidate IDs, scalar losses, and learned assignment records.
    """
    shared_strategy = initialize_fairfeddrift_runtime(servers)
    if shared_strategy.parameters is None:
        raise ValueError("FairFedDrift runtime requires resolved strategy parameters.")
    threshold = shared_strategy.parameters['fairfeddrift_loss_threshold']
    client_ids = [client.client_id for client in clients]
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("FairFedDrift decision clients must have unique IDs.")

    candidate_models = tuple((cluster_id, shared_strategy.cluster_models[cluster_id])
                             for cluster_id in shared_strategy.cluster_models)
    if decision_loaders is None:
        decision_loaders = {client.client_id: prepare_fairfeddrift_decision_loader(client)
                            for client in clients}
    elif not isinstance(decision_loaders, dict) or set(decision_loaders) != set(client_ids):
        raise ValueError("FairFedDrift decision loaders must contain exactly one entry per client.")
    candidate_losses = {}
    for client in clients:
        loader = decision_loaders[client.client_id]
        candidate_losses[client.client_id] = {
            cluster_id: evaluate_fairfeddrift_loss(model, loader)
            for cluster_id, model in candidate_models
        }

    records = []
    for client in clients:
        assigned_cluster_id, created = shared_strategy.assign_client(
            client.client_id, candidate_losses[client.client_id], threshold)
        records.append({
            'client_id': client.client_id,
            'candidate_cluster_ids': tuple(cluster_id for cluster_id, _ in candidate_models),
            'candidate_losses': dict(candidate_losses[client.client_id]),
            'assigned_cluster_id': assigned_cluster_id,
            'created_cluster': created
        })

    active_cluster_ids = synchronize_fairfeddrift_servers(servers, shared_strategy)
    link_clients_to_fairfeddrift_servers(servers, clients, active_cluster_ids,
                                         shared_strategy.client_assignments)
    return records


def prepare_fairfeddrift_timestep(servers: List[Server], clients: List[Client],
                                  round_idx: int) -> Dict[str, Any]:
    """
    Advance retained history each round and make one FairFedDrift decision per data timestep.
    The current timestep's captured data is assigned and stored immediately, but excluded
    from its own historical training loaders until a later timestep.
    :param servers: Mutable flat leaf-server list with FairFedDrift active.
    :param clients: All clients with current ordinary training loaders.
    :param round_idx: Current communication-round index.
    :return: Decision records, per-client historical loaders/counts and active server models.
    """
    if isinstance(round_idx, bool) or not isinstance(round_idx, int) or round_idx < 0:
        raise ValueError("FairFedDrift timestep round must be a non-negative integer.")
    shared_strategy = initialize_fairfeddrift_runtime(servers)
    if shared_strategy.parameters is None:
        raise ValueError("FairFedDrift runtime requires resolved strategy parameters.")
    parameters = shared_strategy.parameters
    rounds_per_timestep = parameters['fairfeddrift_rounds_per_timestep']
    timestep_key = round_idx // rounds_per_timestep
    client_ids = [client.client_id for client in clients]
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("FairFedDrift timestep clients must have unique IDs.")

    histories = shared_strategy.client_data_histories
    for client_id in client_ids:
        histories.setdefault(client_id, ClientDataHistory(parameters['fairfeddrift_window']))
    for client_id, history in histories.items():
        expired_rounds = history.advance(round_idx)
        assignments = shared_strategy.assignment_history.get(client_id, {})
        for expired_round in expired_rounds:
            assignments.pop(expired_round, None)
        if not assignments:
            shared_strategy.assignment_history.pop(client_id, None)

    new_timestep = timestep_key != shared_strategy.last_timestep_key
    decision_records = []
    merge_records = []
    current_data_sample_counts = {}
    if new_timestep:
        decision_loaders = {client.client_id: prepare_fairfeddrift_decision_loader(client)
                            for client in clients}
        current_data_sample_counts = {
            client_id: len(loader.dataset) for client_id, loader in decision_loaders.items()
        }
        merge_records = shared_strategy.merge_clusters(
            histories, round_idx, parameters['fairfeddrift_loss_threshold'])
        decision_records = run_fairfeddrift_decisions(servers, clients, decision_loaders)
        for client in clients:
            client_id = client.client_id
            histories[client_id].add(round_idx, decision_loaders[client_id].dataset)
            shared_strategy.record_assignment(client_id, round_idx)
        shared_strategy.last_timestep_key = timestep_key
        shared_strategy.current_timestep_start_round = round_idx
    else:
        cluster_ids = synchronize_fairfeddrift_servers(servers, shared_strategy)
        link_clients_to_fairfeddrift_servers(servers, clients, cluster_ids,
                                             shared_strategy.client_assignments)

    timestep_start_round = shared_strategy.current_timestep_start_round
    per_client_loaders = {}
    per_client_sample_counts = {}
    for client in clients:
        client_id = client.client_id
        loaders, sample_counts = build_fairfeddrift_history_loaders(
            histories[client_id], shared_strategy.assignment_history.get(client_id, {}),
            client.mini_batch_size, before_round=timestep_start_round)
        per_client_loaders[client_id] = loaders
        per_client_sample_counts[client_id] = sample_counts

    server_models = {server.fairfeddrift_cluster_id: server.model for server in servers}
    if len(server_models) != len(servers):
        raise ValueError("FairFedDrift timestep has duplicate active server cluster IDs.")
    shared_strategy.runtime_history.append({
        'round': round_idx,
        'timestep_id': timestep_key,
        'timestep_start_round': timestep_start_round,
        'new_timestep': new_timestep,
        'current_data_sample_counts': current_data_sample_counts,
        'decisions': [
            dict(record, round=round_idx, timestep_id=timestep_key,
                 data_sample_count=current_data_sample_counts[record['client_id']])
            for record in decision_records
        ],
        'merges': [dict(record, round=round_idx, timestep_id=timestep_key)
                   for record in merge_records],
        'active_cluster_ids': list(server_models),
        'client_assignments': dict(shared_strategy.client_assignments),
        'history_upload_sample_counts': copy.deepcopy(per_client_sample_counts)
    })
    return {
        'new_timestep': new_timestep,
        'decision_records': decision_records,
        'merge_records': merge_records,
        'history_loaders': per_client_loaders,
        'history_sample_counts': per_client_sample_counts,
        'server_models': server_models,
        'timestep_start_round': timestep_start_round
    }


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
            if getattr(getattr(server, 'strategy', None), 'strategy_name', None) == \
                    constants.RecoveryAlgorithm.FAIRFEDDRIFT:
                record['fairfeddrift_cluster_id'] = server.fairfeddrift_cluster_id
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


def evaluate_fedbabu_personalized_clients(all_clients: List[Client], servers: List[Server],
                                          round_idx: int) -> Dict:
    """
    Evaluate each final FedBABU personalized model through its client's test loader.
    :param all_clients: All clients, returned in stable client-ID order.
    :param servers: Single-level FedBABU server list used to record server identity.
    :param round_idx: Final federated round index.
    :return: One structured post-training personalized evaluation record.
    """
    if len(servers) != 1:
        raise ValueError("FedBABU personalized evaluation currently requires one flat server.")
    server = servers[0]
    records = []
    client_ids = [client.client_id for client in all_clients]
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("FedBABU personalized evaluation requires unique client IDs.")

    for client in sorted(all_clients, key=lambda item: item.client_id):
        if client.parent_server_id != server.server_id:
            raise ValueError("FedBABU client is not assigned to the single final server.")
        loss, accuracy = client.evaluate_fedbabu_personalized()
        records.append({
            'client_id': client.client_id,
            'parent_server_id': server.server_id,
            'server_abs_id': server.abs_id,
            'model_id': 'personalized',
            'model_role': 'fedbabu_personalized',
            'loss': loss,
            'accuracy': accuracy
        })

    return {'round': round_idx, 'stage': 'post_training_personalization', 'clients': records}


def build_fedbabu_personalized_log(evaluation_record: Dict) -> Dict:
    """
    Wrap one FedBABU final evaluation record in the repository structured-log envelope.
    :param evaluation_record: Final client-level personalized evaluation record.
    :return: Versioned log payload ready for `write_structured_log()`.
    """
    if not isinstance(evaluation_record, dict) or \
            evaluation_record.get('stage') != 'post_training_personalization':
        raise ValueError("FedBABU personalized log requires its post-training evaluation record.")
    clients = evaluation_record.get('clients')
    if not isinstance(clients, list) or not clients:
        raise ValueError("FedBABU personalized log requires at least one client result.")
    if any(not isinstance(client, dict) or client.get('model_role') != 'fedbabu_personalized'
           for client in clients):
        raise ValueError("FedBABU personalized log records must identify the personalized model role.")
    return {'schema_version': 1, 'records': [copy.deepcopy(evaluation_record)]}


def finalize_fedbabu_personalization(all_clients: List[Client], servers: List[Server],
                                    final_round: int) -> Dict:
    """
    Build, fine-tune and evaluate each client's final FedBABU personal model.
    :param all_clients: Clients participating in the FedBABU simulation.
    :param servers: Single-level FedBABU server list containing the final model.
    :param final_round: Index of the final federated aggregation round.
    :return: Structured final personalized evaluation record.
    """
    initialize_fedbabu_personal_models(all_clients, servers)
    for client in sorted(all_clients, key=lambda item: item.client_id):
        client.fine_tune_fedbabu_personal_head()
    return evaluate_fedbabu_personalized_clients(all_clients, servers, final_round)


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


def build_fairfeddrift_state_log(shared_strategy: Any) -> Dict:
    """
    Build a scalar-only FairFedDrift record with settings and retained runtime events.
    :param shared_strategy: Shared FairFedDrift strategy instance for the simulation.
    :return: Versioned strategy state and per-round decision/history records.
    """
    if shared_strategy.strategy_name != constants.RecoveryAlgorithm.FAIRFEDDRIFT:
        raise ValueError("FairFedDrift logging requires the FairFedDrift strategy instance.")
    parameters = shared_strategy.parameters
    if not isinstance(parameters, dict):
        raise ValueError("FairFedDrift logging requires resolved strategy parameters.")
    return {
        'schema_version': 1,
        'strategy': constants.RecoveryAlgorithm.FAIRFEDDRIFT,
        'adaptation': 'single_group_client_drift',
        'parameters': copy.deepcopy(parameters),
        'events': copy.deepcopy(shared_strategy.runtime_history),
        'final_state': {
            'active_cluster_ids': list(shared_strategy.cluster_models),
            'client_assignments': dict(shared_strategy.client_assignments),
            'previous_losses': dict(shared_strategy.previous_losses),
            'retained_assignment_history': {
                client_id: dict(assignments)
                for client_id, assignments in shared_strategy.assignment_history.items()
            }
        }
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

    fairfeddrift_active = (bool(servers) and
                           servers[0].strategy.strategy_name == constants.RecoveryAlgorithm.FAIRFEDDRIFT)
    fairfeddrift_timestep = None
    if fairfeddrift_active:
        fairfeddrift_timestep = prepare_fairfeddrift_timestep(servers, all_clients, drift.current_round)

    for client in all_clients:
        # Get the server to which the client is connected
        server = servers[client.parent_server_id]
        client.fairfeddrift_training_cluster_id = (
            getattr(server, 'fairfeddrift_cluster_id', server.server_id)
            if client.client_id in sampled_client_ids and
            (fairfeddrift_active or getattr(server, 'fairfeddrift_strategy', None) is not None) else None)

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

        if fairfeddrift_active:
            if client.client_id in sampled_client_ids:
                client.fit_fairfeddrift_history(
                    fairfeddrift_timestep['server_models'],
                    fairfeddrift_timestep['history_loaders'][client.client_id],
                    fairfeddrift_timestep['history_sample_counts'][client.client_id])
            else:
                client.fit_fairfeddrift_history({}, {}, {})

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
        if drift_recovery_parameters['recovery_method'] != constants.RecoveryAlgorithm.FEDRC:
            # Ditto is run-wide; other methods restore the configured base strategy after drift.
            after_drift_method = drift_recovery_parameters['base_aggregation_method']
            if drift_recovery_parameters['recovery_method'] == constants.RecoveryAlgorithm.DITTO:
                after_drift_method = constants.RecoveryAlgorithm.DITTO
            if (after_drift_method == constants.RecoveryAlgorithm.ORACLE and
                    getattr(drift, 'fairfeddrift_parked_base_servers', None) is not None):
                server_hierarchy[-1][:] = drift.fairfeddrift_parked_base_servers
                del drift.fairfeddrift_parked_base_servers
            change_server_aggregation_strategy(server_hierarchy, after_drift_method, drift,
                                               drift_recovery_parameters)
            if after_drift_method == constants.RecoveryAlgorithm.ORACLE and not drift.is_synchronous:
                link_clients_to_servers_by_drift_id(clients, server_hierarchy[-1])

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
        # Activate the configured recovery strategy for the drift phase.
        if (drift_recovery_parameters['recovery_method'] == constants.RecoveryAlgorithm.FAIRFEDDRIFT and
                drift_recovery_parameters['base_aggregation_method'] == constants.RecoveryAlgorithm.ORACLE):
            # FairFedDrift changes the flat leaf-server list as learned clusters appear/merge.
            # Retain Oracle's ground-truth servers so the configured base can resume after drift.
            drift.fairfeddrift_parked_base_servers = list(server_hierarchy[-1])
        change_server_aggregation_strategy(server_hierarchy, drift_recovery_parameters['recovery_method'],
                                           drift, drift_recovery_parameters)

        # Change the clients' recovery method for the existing triggered strategies.
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
