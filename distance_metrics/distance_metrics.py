"""
Description: This file consists the Averaging functions for FedAvg algorithm.

Author: Nisal Hemadasa
Date: 26-04-2025
Version: 1.0
"""

import torch
from collections import OrderedDict
from typing import List, Dict, Tuple


def compute_euclidean_distance_weights(prev_edge_model: OrderedDict,
                                       client_model_params_list: List[OrderedDict]) -> Tuple[
                                        List[float], List[float], Dict[int, Dict[str, float]]]:
    """
    Computes normalized inverse-euclidean-distance weights for client models relative to the previous edge model.
    :param: prev_edge_model: State dict of the previous global model at the edge server
    :param: client_model_params_list: List of state dicts of the client models
    :return: weights: List of normalized inverse-euclidean-distance weights for each client model,
            client_model_distances: List of distances,
            client_layer_distances: Dictionary of distances of each separate layer from  the corresponding layer in the
            edge model for each client model
    """
    client_model_distances = []
    client_layer_distances = {}

    for index, client_params in enumerate(client_model_params_list):
        total_distance = 0.0
        layer_distance = {}

        for key in prev_edge_model.keys():
            if key in client_params and prev_edge_model[key].shape == client_params[key].shape:
                diff = client_params[key] - prev_edge_model[key]

                # Compute the L2 (Euclidean) distance for this layer
                l2_distance = diff.norm(p=2)
                total_distance += l2_distance.item()
                layer_distance[key] = l2_distance.item()

        client_model_distances.append(total_distance)
        client_layer_distances[index] = layer_distance

    distances = torch.tensor(client_model_distances)

    # Normalize using direct distances, as per HAF-Edge Equation (4)
    if torch.sum(distances) == 0:
        # Edge case: if all distances are zero, assign equal weights
        weights = torch.ones_like(distances) / len(distances)
    else:
        weights = distances / distances.sum()

    return weights.tolist(), client_model_distances, client_layer_distances

