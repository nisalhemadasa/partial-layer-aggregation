"""
Description: This file consists the sever side aggregation operations for FedAU algorithm.

Author: Nisal Hemadasa
Date: 01-11-2025
Version: 1.0
"""
import copy
from typing import OrderedDict, Dict, List

import torch
from torch import nn

import constants
from models.model import split_to_extractor_and_classifier, set_parameters


class FedAU:
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    def aggregate_models(self, server_model: nn.Module, client_model_params_dict: Dict[str, OrderedDict],
                         auxiliary_classifier_params_dict: Dict[str, OrderedDict], ema_weight: float) -> None:
        """
        Aggregate the client models to the global model using adaptive weights and returns the new aggregated model.
        :param server_model: The server (edge or global) model
        :param client_model_params_dict: List of state dicts of the client models
        :param auxiliary_classifier_params_dict: List of state dicts of the auxiliary classifiers from drifted clients
        :param ema_weight: Exponential moving average weight: alpha
        :return: unlearning_composite_params: The parameters of aggregated composite unlearning model {E, W_hat}
        """
        _learning_classifier_params_list = []
        _auxiliary_classifier_params_list = []
        extractors_params_list = []

        for client_id, client_model_params in client_model_params_dict.items():
            # To all clients: split learning model to extractor and classifier parameters
            extractor_params, _learning_classifier_params = split_to_extractor_and_classifier(None,
                                                                                              client_model_params_dict[
                                                                                                  client_id])
            if auxiliary_classifier_params_dict[client_id] is not None:  # Drifted clients

                # Get auxiliary classifier parameters of the drifted client
                auxiliary_classifier_params = auxiliary_classifier_params_dict[client_id]

                _auxiliary_classifier_params_list.append(auxiliary_classifier_params)

            _learning_classifier_params_list.append(_learning_classifier_params)
            extractors_params_list.append(extractor_params)

        # Perform FedAvg
        # on learning classifiers (W_l) of all clients
        fedavg_learning_classifier_params = average_model_parameters(_learning_classifier_params_list)
        # on auxiliary classifiers (W_a) of drifted clients
        fedavg_auxiliary_classifier_params = average_model_parameters(_auxiliary_classifier_params_list)
        # on extractors (E) of all clients
        fedavg_extractor_params = average_model_parameters(extractors_params_list)

        # Exponential Mean Averaging of learning and auxiliary classifiers to get unlearning_model (W_hat)
        # W_hat = alpha * FedAvg(W_l) + (1-alpha) * FedAvg(W_a)
        unlearning_classifier_params = get_exponential_moving_average(fedavg_learning_classifier_params,
                                                                      fedavg_auxiliary_classifier_params, ema_weight)

        # Append the unlearning_model to extractor ({E, W_hat})
        unlearning_composite_params = OrderedDict(
            list(fedavg_extractor_params.items()) + list(unlearning_classifier_params.items()))

        # Load the composite unlearning model ({E, W_hat}) to the server model
        set_parameters(server_model, unlearning_composite_params)


def get_exponential_moving_average(learning_params: OrderedDict, aux_params: OrderedDict,
                                   _ema_weight: float) -> OrderedDict:
    """
    Get the exponential moving average of two sets of parameters
    :param learning_params: First set of parameters (learning module classifier parameters)
    :param aux_params: Second set of parameters (auxiliary classifier parameters)
    :param _ema_weight: averaging weights
    :return averaged_params: Averaged parameters
    """
    averaged_params = OrderedDict()
    for key in learning_params.keys():
        averaged_params[key] = _ema_weight * learning_params[key] + (1 - _ema_weight) * aux_params[key]
    return averaged_params


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
    _strategy = FedAU(strategy_name=constants.RecoveryAlgorithm.FEDAU)
    return _strategy
