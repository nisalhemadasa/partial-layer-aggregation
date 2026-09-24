"""Synthetic data for FairFedDrift tests only; experiments use main.py drift settings."""
import torch
from torch.utils.data import TensorDataset


def make_client_drift_fixture():
    """
    Build independent before/after datasets for six clients and three concepts.
    :return: Before datasets, after datasets and evaluation-only drift IDs by client.
    """
    labels = torch.arange(10).repeat(2)
    inputs = torch.eye(10)[labels]
    # Ground truth is returned separately and never attached to client datasets.
    drift_ids = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
    mappings = {
        0: torch.arange(10),
        1: torch.tensor([0, 2, 1, 4, 3, 5, 6, 7, 8, 9]),
        2: torch.tensor([0, 1, 2, 3, 4, 7, 6, 5, 8, 9]),
    }
    before = {}
    after = {}
    for client_id, drift_id in drift_ids.items():
        before[client_id] = TensorDataset(inputs.clone(), labels.clone())
        after[client_id] = TensorDataset(inputs.clone(), mappings[drift_id][labels].clone())
    return before, after, drift_ids
