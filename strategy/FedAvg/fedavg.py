"""
Description: This file consists the Averaging functions for FedAvg algorithm.

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
from typing import List, OrderedDict

import torch

from distance_metrics.distance_metrics import compute_euclidean_distance_weights
from models.model import CNNModel


class FedAvg:
    def __init__(self):
        pass

    def aggregate_models(self, model: CNNModel, client_model_params_list: List[OrderedDict]) -> CNNModel:
        """ Aggregate the client models to the global model and returns the new aggregated model
        :param model: The server (edge or global) model
        :param client_model_params_list: List of state dicts of the client models
        :return: model: The updated server (edge or global)  model after aggregation
        """
        model_params = model.state_dict()

        # Simple averaging of weights
        updated_model_params = model_params.copy()
        for i in updated_model_params.keys():
            updated_model_params[i] = torch.stack(
                [client_model_params[i].float() for client_model_params in client_model_params_list], 0).mean(0)

        model.load_state_dict(updated_model_params)
        return model


def aggregator_fn():
    """ Returns an instance of the FedAvg aggregation strategy """
    _strategy = FedAvg()
    return _strategy
