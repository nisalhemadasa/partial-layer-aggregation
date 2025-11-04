"""
Description: This file consists the sever side aggregation operations for FedAU algorithm.

Author: Nisal Hemadasa
Date: 01-11-2025
Version: 1.0
"""
import copy
from typing import OrderedDict, Dict, List

import torch

from models.model import CNNModel, split_to_extractor_and_classifier
from strategy.FedAvg import fedavg


class FedAU:
    def __init__(self):
        pass

    def aggregate_models(self, server_model: CNNModel, client_model_params_dict: Dict[str, OrderedDict],
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
            if client_id in auxiliary_classifier_params_dict:  # Drifted clients

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
        server_model.load_state_dict(unlearning_composite_params)


def aggregator_fn():
    """ Returns an instance of the FedAvg aggregation strategy """
    _strategy = FedAU()
    return _strategy


def get_exponential_moving_average(params_1: OrderedDict, params_2: OrderedDict,
                                   _ema_weight: float) -> OrderedDict:
    """
    Get the exponential moving average of two sets of parameters
    :param params_1: First set of parameters
    :param params_2: Second set of parameters
    :param _ema_weight: averaging weights
    :return averaged_params: Averaged parameters
    """
    # Make a deep copy so we don't modify originals
    averaged_params = copy.deepcopy(params_1)
    for key in params_1.keys():
        averaged_params[key] = _ema_weight * params_1[key] + (1 - _ema_weight) * params_2[key]
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
