"""
Description: This file consists supporting functions for FedRC (Robust Clustering) algorithm.

Y. Guo, X. Tang, and T. Lin, “FedRC: Tackling Diverse Distribution Shifts Challenge in Federated Learning by Robust
Clustering,” in Proceedings of the 41st International Conference on Machine Learning (ICML 2024), 2024.

Author: Nisal Hemadasa
Date: 04-12-2025
Version: 1.0
"""
from typing import List

import torch

from federated_network.client import Client


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

