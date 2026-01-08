"""
Description: This file consists supporting functions for Oracle (clustering based multi-global model) algorithm.

Author: Nisal Hemadasa
Date: 08-01-2026
Version: 1.0
"""
from typing import List, Dict, OrderedDict

import torch
from torch import nn

import constants

DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__}"
)


class Oracle:
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    def aggregate_models(self, oracle_server_models: List[nn.Module],
                         client_model_params_dict: Dict[str, List[OrderedDict]]) -> None:
        """
        Aggregate the client models to the corresponding global model (out of the multiple available global models) and
        returns the new aggregated model.
        *** Here we use FedAvg in each global model.***
        :param oracle_server_models: The list of models (cluster bases) in the server model
        :param client_model_params_dict: Dictionary containing the server IDs (keys) and the corresponding state dicts
        of the client models
        return: None
        """
        for idx in range(len(oracle_server_models)):
            server_model_params = oracle_server_models[idx].state_dict()

            if client_model_params_dict:  # if not empty
                client_model_params_list = client_model_params_dict.values()

            # Simple averaging of weights
            updated_server_model_params = server_model_params.copy()
            for key in updated_server_model_params.keys():
                updated_server_model_params[key] = torch.stack(
                    [client_model_params[idx][key].float() for client_model_params in client_model_params_list],
                    0).mean(0)

            # Load the FedAvg-ed parameters to the server model
            set_parameters(fedrc_server_models[idx], updated_server_model_params)


def aggregator_fn():
    """ Returns an instance of the FedAvg aggregation strategy """
    _strategy = Oracle(strategy_name=constants.RecoveryAlgorithm.FEDRC)
    return _strategy
