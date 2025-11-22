"""
Description: This file consists the sever side aggregation operations for FedAU algorithm.

Author: Nisal Hemadasa
Date: 01-11-2025
Version: 1.0
"""
from typing import OrderedDict, Dict, List

import torch
from torch import nn

import constants
from models.model import CNNModel, split_to_extractor_and_classifier, set_parameters


class FedEx:
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    def aggregate_models(self, server_model: CNNModel, client_model_params_dict: Dict[str, OrderedDict]) -> OrderedDict:
        """
        Aggregate the client models to the global model using adaptive weights and returns the new aggregated model.
        :param server_model: The server (edge or global) model
        :param client_model_params_dict: List of state dicts of the client models
        :return: unlearning_composite_params: The parameters of aggregated composite unlearning model {E, W_hat}
        """
        extractors_params_list = []

        for client_id, client_model_params in client_model_params_dict.items():
            # To all clients: split learning model to extractor and classifier parameters
            extractor_params, _learning_classifier_params = split_to_extractor_and_classifier(None,
                                                                                              client_model_params_dict[
                                                                                                  client_id])
            extractors_params_list.append(extractor_params)

        # Perform FedAvg on extractors (E) of all clients
        fedavg_extractor_params = average_model_parameters(extractors_params_list)

        # TODO: testing
        # Replace fc2 (last) layer with a new nn.Linear of the same shape
        server_model.fc2 = nn.Linear(server_model.fc2.in_features, server_model.fc2.out_features)

        # Append the unlearning_model to extractor ({E, W_hat})
        unlearning_composite_params = OrderedDict(
            list(fedavg_extractor_params.items()) + list(server_model.fc2.items()))

        set_parameters(server_model, unlearning_composite_params)


def average_model_parameters(model_params_list: List[OrderedDict]) -> OrderedDict:
    """
    Average a list of model parameters
    :param model_params_list: List of model parameters
    :return averaged_params: Averaged model parameters
    """
    model_keys = model_params_list[0].keys()

    # Simple averaging of weights
    averaged_model_params = model_params_list[0].copy()
    for i in model_keys:
        averaged_model_params[i] = torch.stack(
            [model_params[i].float() for model_params in model_params_list], 0).mean(0)

    return averaged_model_params


def aggregator_fn():
    """ Returns an instance of the FedAvg aggregation strategy """
    _strategy = FedEx(strategy_name=constants.RecoveryAlgorithm.FEDEX)
    return _strategy
