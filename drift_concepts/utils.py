"""
Description: This script contains helper functions related to drift generation.

Author: Nisal Hemadasa
Date: 09-01-2026
Version: 1.0
"""
from typing import Dict, Any

import torch


def get_clients_indices_with_drift(_num_client_instances: int, _num_drifted_clients: int, is_synchronous: bool,
                                   is_random: bool) -> list[Any] | None:
    """
    Get the list of clients that have drifted data. This function is only applicable for the synchronous drift case.
    :param _num_client_instances: Total number of client instances in the federated network
    :param _num_drifted_clients: Number of client undergoing drift
    :param is_synchronous: Boolean indicating if the drift is synchronous or asynchronous
    :param is_random: Boolean indicating if the drifted clients are selected randomly at the beginning
    :return: Indices of clients with drifted data
    """
    if is_random and is_synchronous:  # only for the synchronous drift case
        # get random permutation of client indices undergoing drift
        client_indices = torch.randperm(_num_client_instances).tolist()
        drifted_clients_indices = client_indices[:_num_drifted_clients]
        return drifted_clients_indices
    else:
        return None


def cluster_client_indices_by_drift_patterns(_num_client_instances: int, _num_drifted_clients: int,
                                             drift_group_sizes: list[list[int]], is_synchronous: bool,
                                             async_drift_specs: Dict) -> list[list[int]]:
    """
    Arrange client indices into groups (clusters) based on their drift patterns, w.r.t each drift timestep.
    :param _num_client_instances: Total number of client instances in the federated network
    :param _num_drifted_clients: Number of client undergoing drift
    :param drift_group_sizes: Sizes of the drift affected client groups at each drift_step_rounds
        - outer list : timesteps (len(drift_group_sizes) -> number of timesteps)
        - inner list : sizes of drift groups (len(drift_group_sizes[0]) ->sequence of drifted client group sizes)
    :param is_synchronous: Boolean indicating if the drift is synchronous or asynchronous
    :param async_drift_specs: Dictionary containing the specifications for asynchronous drift
    :return: Clustered client indices based on their drift patterns.
        - outer list : timesteps (len(drift_group_sizes) -> number of timesteps)
        - inner list : drift groups. i.e. client indices grouped together based on their drift patterns.

    """
    if not is_synchronous:  # only for asynchronous drift cases
        # get the first N clients as undergoing drift
        drift_clustered_client_indices = []

        for drift_timestep in drift_group_sizes:
            current_group = []
            client_idx = 0  # restart counting for each timestep

            for group_size in drift_timestep:
                # get consecutive client indices for the current drift group
                current_group.append(list(range(client_idx, client_idx + group_size)))
                client_idx += group_size

            drift_clustered_client_indices.append(current_group)

        async_drift_specs['drift_groups'] = drift_clustered_client_indices
        return drift_clustered_client_indices
    else:
        raise ValueError("Synchronous drift case is not implemented yet.")


def compose_label_mapping(mapping, class_pairs):
    """
    Compose ordered whole-class swaps onto a current label permutation.
    :param mapping: Tuple mapping each original class to its current label.
    :param class_pairs: Ordered pairs of current labels to exchange.
    :return: Updated immutable label mapping.
    """
    if (not isinstance(mapping, tuple) or not mapping or
            any(type(label) is not int for label in mapping) or
            set(mapping) != set(range(len(mapping)))):
        raise ValueError('Label mapping must be a permutation of the complete class domain.')
    for pair in class_pairs:
        if (len(pair) != 2 or any(type(label) is not int or label not in range(len(mapping))
                                  for label in pair)):
            raise ValueError('Swap pairs must contain two labels from the class domain.')
        left, right = pair
        mapping = tuple(right if value == left else left if value == right else value for value in mapping)
    return mapping


def reachable_label_mappings(drift, client_ids, num_classes):
    """
    Enumerate mappings reached by a finite asynchronous whole-class swap schedule.
    :param drift: Configured drift with client groups and operation patterns.
    :param client_ids: IDs of all clients, including clients without drift.
    :param num_classes: Complete dataset class count.
    :return: Stable tuple of reachable signatures, beginning with identity.
    """
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError('Oracle requires a positive complete class count.')
    identity = tuple(range(num_classes))
    mappings = dict.fromkeys(client_ids, identity)
    if not mappings or len(mappings) != len(client_ids):
        raise ValueError('Oracle requires nonempty, unique client IDs.')
    groups = drift.drift_clustered_client_indices
    operations = drift.drift_patterns_over_time
    if len(groups) != len(operations):
        raise ValueError('Oracle drift groups and operation steps must match.')
    for operation_id, pairs in drift.drift_pattern_id_map.items():
        if operation_id == 0:
            raise ValueError('Operation ID zero must remain the no-drift ID.')
        compose_label_mapping(identity, pairs)
    reached = {identity: None}
    for step_groups, step_operations in zip(groups, operations):
        if len(step_groups) != len(step_operations):
            raise ValueError('Each drift group needs exactly one operation ID.')
        assigned = set()
        for group, operation_id in zip(step_groups, step_operations):
            if operation_id != 0 and operation_id not in drift.drift_pattern_id_map:
                raise ValueError('Unknown Oracle drift operation ID.')
            for client_id in group:
                if client_id not in mappings or client_id in assigned:
                    raise ValueError('Drift groups contain unknown or repeated client IDs.')
                assigned.add(client_id)
                mappings[client_id] = compose_label_mapping(
                    mappings[client_id], drift.drift_pattern_id_map.get(operation_id, ()))
        for mapping in mappings.values():
            reached.setdefault(mapping, None)
    return tuple(reached)
