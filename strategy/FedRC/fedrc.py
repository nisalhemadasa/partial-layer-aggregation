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

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__}"
)


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


def compute_omega(_clients: List[Client]):
    for client in _clients:
        fedrc_models = client.fedrc_models  # List[nn.Module], length K
        num_clusters = len(fedrc_models)

        prev_omega = client.omega_i_k.to(DEVICE)  # [K]
        prev_C = client.C_y_k.to(DEVICE)  # [num_classes, K]

        # Accumulators over all local samples j
        sum_gamma_per_cluster = torch.zeros(num_clusters, device=DEVICE)  # ∑_j γ_{i,j;k}
        label_gamma = torch.zeros(client.num_classes, num_clusters, device=DEVICE)  # ∑_j 1{y_j=y} γ_{i,j;k}
        num_samples = 0

        # Loop over local data
        # In the paper, one sample at a time is used, but here we use minibatches for efficiency (faster on GPU), but
        # is mathematically equivalent (just vectorized).
        for x, y in client.trainloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)  # B = batch size -> Label vector y is a 1D tensor of length [B]. This is equal to the 'minibatch_size' given in network.py.
            batch_size = y.size(0)  # y.size(0) == B
            num_samples += batch_size

            # Compute per-sample loss for each cluster model
            # losses_all: [B, K]
            losses_all = torch.empty(batch_size, num_clusters, device=DEVICE)

            for k, model_k in enumerate(fedrc_models):
                model_k.to(DEVICE)
                model_k.eval()  # we only evaluate here

                with torch.no_grad():
                    logits = model_k(x)  # [B, num_classes]
                    # per-sample cross-entropy loss
                    losses_all[:, k] = F.cross_entropy(
                        logits, y, reduction="none"
                    )


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
