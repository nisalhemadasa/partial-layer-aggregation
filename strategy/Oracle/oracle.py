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
from strategy.FedAvg import FedAvg

DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__}"
)


class Oracle:
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

        # Inside the clusters, Oracle uses FedAvg to aggregate its cluster-member client models
        self.fedavg = FedAvg(constants.RecoveryAlgorithm.FEDAVG)

    def aggregate_models(self, server_model: nn.Module,
                         client_model_params_dict: Dict[str, OrderedDict]) -> None:
        """
        Aggregate the client models to the corresponding global model (out of the multiple available global models) and
        returns the new aggregated model.
        *** Here we use FedAvg in each global model.***
        :param server_model: One of multiple server models in the Oracle server cluster
        :param client_model_params_dict: Dictionary containing the server IDs (keys) and the corresponding state dicts
        of the client models
        return: None
        """
        self.fedavg.aggregate_models(server_model, client_model_params_dict, None)


def aggregator_fn():
    """ Returns an instance of the FedAvg aggregation strategy """
    _strategy = Oracle(strategy_name=constants.RecoveryAlgorithm.ORACLE)
    return _strategy
