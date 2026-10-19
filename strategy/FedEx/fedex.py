"""
Description: This file consists the sever side aggregation operations for FedAU algorithm.

Author: Nisal Hemadasa
Date: 01-11-2025
Version: 1.0
"""
from typing import OrderedDict, Dict, List

import numpy as np
import torch
from torch import nn

import constants
from models.utils import split_to_extractor_and_classifier, set_parameters
from strategy.FedAvg import FedAvg


class FedEx:
    def __init__(self, strategy_name: str, shared_layer_count: int = None):
        self.strategy_name = strategy_name
        if shared_layer_count is not None and (
                isinstance(shared_layer_count, bool) or
                not isinstance(shared_layer_count, int) or
                not 1 <= shared_layer_count <= 4):
            raise ValueError("shared_layer_count must be an integer from 1 through 4.")
        self.shared_layer_count = shared_layer_count

        # Inside the clusters, Oracle uses FedAvg to aggregate its cluster-member client models
        self.fedavg = FedAvg(constants.RecoveryAlgorithm.FEDAVG)

    def aggregate_models(self, server_model: nn.Module,
                         client_model_params_dict: Dict[str, OrderedDict]) -> OrderedDict:
        """
        Aggregate the client models to the global model using adaptive weights and returns the new aggregated model.
        :param server_model: The server (edge or global) model
        :param client_model_params_dict: List of state dicts of the client models
        :return: unlearning_composite_params: The parameters of aggregated composite unlearning model {E, W_hat}
        """
        extractors_params_list = []

        # self.fedavg.aggregate_models(server_model, client_model_params_dict, None)

        for client_id, client_model_params in client_model_params_dict.items():
            # To all clients: split learning model to extractor and classifier parameters
            extractor_params, _ = split_to_extractor_and_classifier(
                None, client_model_params_dict[client_id], server_model.get_model_type(),
                shared_layer_count=self.shared_layer_count)
            extractors_params_list.append(extractor_params)

        # Perform FedAvg on extractors (E) of all clients
        fedavg_extractor_params = average_model_parameters(extractors_params_list)

        set_parameters(server_model, fedavg_extractor_params, False)


def average_model_parameters(model_params_list: List[OrderedDict]) -> OrderedDict:
    """
    Average a list of model parameters
    :param model_params_list: List of model parameters
    :return averaged_params: Averaged model parameters
    """
    model_keys = model_params_list[0].keys()

    # Simple averaging of weights
    averaged_model_params = model_params_list[0].copy()  # TODO: what you add here server model params?
    for i in model_keys:
        averaged_model_params[i] = torch.stack(
            [model_params[i].float() for model_params in model_params_list], 0).mean(0)

    return averaged_model_params


def aggregator_fn(shared_layer_count: int = None):
    """Create a FedEx aggregator with an optional shared CIFAR-10 layer prefix.

    :param shared_layer_count: Number of leading CIFAR-10 learned layers to aggregate
    :return: FedEx strategy instance
    """
    _strategy = FedEx(strategy_name=constants.RecoveryAlgorithm.FEDEX,
                      shared_layer_count=shared_layer_count)
    return _strategy
