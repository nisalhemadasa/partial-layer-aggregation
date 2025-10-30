"""
Description: This file consists the sever side aggregation operations for FedAU algorithm.

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
from typing import List, OrderedDict

from models.model import CNNModel, split_to_extractor_and_classifier


class FedAU:
    def __init__(self):
        pass

    def aggregate_models(self, model: CNNModel, client_model_params_list: List[OrderedDict],
                         auxiliary_classifier_params: List[OrderedDict]) -> CNNModel:
        """
        Aggregate the client models to the global model using adaptive weights and returns the new aggregated model.
        :param model: The server (edge or global) model
        :param client_model_params_list: List of state dicts of the client models
        :param auxiliary_classifier_params: List of state dicts of the auxiliary classifiers from drifted clients
        :return: model: The updated server (edge or global)  model after aggregation
        """
        # get learning model parameters
        learning_model_params = model.state_dict()

        # split learning model to extractor and classifier parameters
        extractor, _learning_model_classifier_parameters = split_to_extractor_and_classifier(model)


def aggregator_fn():
    """ Returns an instance of the FedAvg aggregation strategy """
    _strategy = FedAU()
    return _strategy