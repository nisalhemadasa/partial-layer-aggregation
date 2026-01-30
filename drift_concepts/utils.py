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