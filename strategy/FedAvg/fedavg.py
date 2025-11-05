"""
Description: This file consists the Averaging functions for FedAvg algorithm.

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
from typing import List, OrderedDict, Dict

import torch

from distance_metrics.distance_metrics import compute_euclidean_distance_weights
from models.model import CNNModel


class FedAvg:
    def __init__(self):
        pass

    def aggregate_models(self, server_model: CNNModel,
                         client_model_params_dict: Dict[str, OrderedDict],
                         client_model_params_list: List[OrderedDict] = None) -> None:
        """
        Aggregate the client models to the global model and returns the new aggregated model
        :param server_model: The server (edge or global) model
        :param client_model_params_dict: Dictionary containing the server IDs (keys) and the corresponding state dicts of the client models
        :param client_model_params_list: List of state dicts of the client models (used in the FedAU implementation)
        :return: The parameters of the server (edge or global) model after aggregation (or averaged layers in FedAU's case)
        """
        server_model_params = server_model.state_dict()

        if client_model_params_dict is not None:
            client_model_params_list = client_model_params_dict.values()

        # Simple averaging of weights
        updated_server_model_params = server_model_params.copy()
        for key in updated_server_model_params.keys():
            updated_server_model_params[key] = torch.stack(
                [client_model_params[key].float() for client_model_params in client_model_params_list], 0).mean(0)

        server_model.load_state_dict(updated_server_model_params)


def aggregator_fn():
    """ Returns an instance of the FedAvg aggregation strategy """
    _strategy = FedAvg()
    return _strategy
