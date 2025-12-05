"""
Description: This file consists supporting functions for FedRC (Robust Clustering) algorithm.

Y. Guo, X. Tang, and T. Lin, “FedRC: Tackling Diverse Distribution Shifts Challenge in Federated Learning by Robust
Clustering,” in Proceedings of the 41st International Conference on Machine Learning (ICML 2024), 2024.

Author: Nisal Hemadasa
Date: 04-12-2025
Version: 1.0
"""
from typing import List, OrderedDict, Dict

import torch
from torch import nn

import constants
from federated_network.client import Client


class FedRC:
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    def aggregate_models(self, fedrc_server_models: List[nn.Module], client_model_params_dict: Dict[str, List[OrderedDict]]) -> None:
        """
        Aggregate the client models to the global model and returns the new aggregated model.
        *** Here we use FedAvg, as the orginal paper allows using any aggregation method ***
        # TODO: This function is not designed to the hierarchical server topology
        :param fedrc_server_models: The list of models (cluster bases) in the server model
        :param client_model_params_dict: Dictionary containing the client IDs (keys) and the corresponding state dicts
        of all the models (fedrc_cluster_count amount of models) adopted by each client
        return: None
        """
        for idx in range(len(fedrc_server_models)):
            server_model_params = fedrc_server_models[idx].state_dict()

            if client_model_params_dict is not None:
                for client in client_model_params_dict.keys():
                    assert len(client_model_params_dict[client]) == len(fedrc_server_models), \
                        "Number of FedRC models in the client and server must be the same."
                    fedrc_client_models = client_model_params_dict[client]
                client_model_params_list = client_model_params_dict.values()

            # Simple averaging of weights
            updated_server_model_params = server_model_params.copy()
            for key in updated_server_model_params.keys():
                updated_server_model_params[key] = torch.stack(
                    [client_model_params[key].float() for client_model_params in client_model_params_list], 0).mean(0)

            # Load the FedAvg-ed parameters to the server model
            set_parameters(server_model, updated_server_model_params)


def get_fedrc_client_model_params(client: Client) -> List[OrderedDict]:
    """
    Return a list of state_dicts for all FedRC models of a client.
    :param client: The client whose FedRC model parameters are to be retrieved
    :return: List of state_dicts for all FedRC models of the client
    """
    return [model.state_dict() for model in client.fedrc_models]


def compute_omega():
    pass


def compute_gamma(loss: float, temperature: float = 1.0) -> torch.Tensor:
    """
    Convert loss vector [K] into responsibilities gamma[i,k] via a softmax over -loss/T.

    Lower loss => higher responsibility.
    """
    losses = torch.tensor(loss, dtype=torch.float32)
    scaled = -losses / temperature
    gamma = torch.softmax(scaled, dim=0)
    return gamma  # shape [K]


def compute_fedrc_metrics(loss: float, clients: List[Client]):
    """
    Compute gamma (data weights) and omega (cluster weights) for FedRC algorithm.
    :param loss: The average Cross entropy loss of all the samples of all participated clients in the current round
    :param clients: List of clients participated in the current round
    """
    compute_gamma(loss)
    compute_omega()


def aggregator_fn():
    """ Returns an instance of the FedAvg aggregation strategy """
    _strategy = FedRC(strategy_name=constants.RecoveryAlgorithm.FEDRC)
    return _strategy
