"""
Description: This file consists the sever side aggregation operations for FedAU algorithm.

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
import copy
from typing import List, OrderedDict

from models.model import CNNModel, split_to_extractor_and_classifier


class FedAU:
    def __init__(self):
        pass

    def aggregate_models(self, server_model: CNNModel, client_model_params_list: List[OrderedDict],
                         auxiliary_classifier_params_list: List[OrderedDict]) -> CNNModel:
        """
        Aggregate the client models to the global model using adaptive weights and returns the new aggregated model.
        :param server_model: The server (edge or global) model
        :param client_model_params_list: List of state dicts of the client models
        :param auxiliary_classifier_params_list: List of state dicts of the auxiliary classifiers from drifted clients
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

        weight_learning = 0.9
        weight_auxiliary = 0.1

        # Get parameters of the server model
        server_model_params = server_model.state_dict()

        # split learning model to extractor and classifier parameters
        for client_model_params in client_model_params_list:
            extractor, _learning_classifier_parameters = split_to_extractor_and_classifier(None, client_model_params)

        # Exponential Mean Averaging of learning and auxiliary classifiers
        averaged_classifier = get_exponential_moving_average(_learning_classifier_parameters, )

        # Append the averaged classifiers to extractor




def aggregator_fn():
    """ Returns an instance of the FedAvg aggregation strategy """
    _strategy = FedAU()
    return _strategy
