"""
Description: This file consists the sever side aggregation operations for FedAU algorithm.

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
import copy
from typing import List, OrderedDict, Dict

from models.model import CNNModel, split_to_extractor_and_classifier
from strategy.FedAvg import fedavg


class FedAU:
    def __init__(self):
        pass

    def aggregate_models(self, server_model: CNNModel, client_model_params_dict: Dict[OrderedDict],
                         auxiliary_classifier_params_dict: Dict[OrderedDict], ema_weight: float) -> CNNModel:
        """
        Aggregate the client models to the global model using adaptive weights and returns the new aggregated model.
        :param server_model: The server (edge or global) model
        :param client_model_params_dict: List of state dicts of the client models
        :param auxiliary_classifier_params_dict: List of state dicts of the auxiliary classifiers from drifted clients
        :param ema_weight: Exponential moving average weight: alpha
        :return: model: The updated server (edge or global)  model after aggregation
        """

        def get_exponential_moving_average(params_1: OrderedDict, params_2: OrderedDict, weight) -> OrderedDict:
            """
            Get the exponential moving average of two sets of parameters
            :param params_1: First set of parameters
            :param params_2: Second set of parameters
            :param weight: averaging weights
            "return avereaged_params: Averaged parameters
            """
            # Make a deep copy so we don't modify originals
            averaged_params = copy.deepcopy(params_1)
            for key in params_1.keys():
                averaged_params[key] = weight * params_1[key] + (1 - weight) * params_2[key]
            return averaged_params

        # Get parameters of the server model
        server_model_params = server_model.state_dict()

        # Create a new FedAvg instance
        fed_avg = fedavg.aggregator_fn()

        _learning_classifier_params_list = []
        _auxiliary_classifier_params_list = []
        extractors_params_list = []
        for client_id, client_model_params in client_model_params_dict.items():
            if client_id in auxiliary_classifier_params_dict:  # Drifted clients
                # split learning model to extractor and classifier parameters
                extractor_params, _learning_classifier_params = split_to_extractor_and_classifier(None,
                                                                                               client_model_params_dict[
                                                                                                   client_id])
                # Get auxiliary classifier parameters of the drifted client
                auxiliary_classifier_params = auxiliary_classifier_params_dict[client_id]

                _learning_classifier_params_list.append(_learning_classifier_params)
                _auxiliary_classifier_params_list.append(auxiliary_classifier_params)
                extractors_params_list.append(extractor_params)

            else:  # Non-drifted clients, perform simple FedAvg
                fedavg_non_drifted_client_params = fed_avg.aggregate_models(client_model_params_dict[client_id])

        # Perform FedAvg on learning classifiers of drifted clients
        fedavg_learning_classifier_params = fed_avg.aggregate_models(None, _learning_classifier_params_list)

        # Perform FedAvg on auxiliary classifiers of drifted clients
        fedavg_auxiliary_classifier_params = fed_avg.aggregate_models(None, _auxiliary_classifier_params_list)

        # Perform FedAvg on extractors of all clients
        fedavg_extractor_params = fed_avg.aggregate_models(None, extractors_params_list)

        # Exponential Mean Averaging of learning and auxiliary classifiers
        averaged_classifier = get_exponential_moving_average(fedavg_learning_classifier_params,
                                                             fedavg_auxiliary_classifier_params, ema_weight)



            # Get learning classifier parameters

        # Append the averaged classifiers to extractor


def aggregator_fn():
    """ Returns an instance of the FedAvg aggregation strategy """
    _strategy = FedAU()
    return _strategy
