"""
Description: This file consists a simple MLP for ADULT dataset classification.

Author: Nisal Hemadasa
Date: 02-04-2026
Version: 1.0
"""

import torch
import torch.nn as nn


class TabularAdultModel(nn.Module):
    """
    Simple MLP for tabular Adult dataset.
    Architecture: Linear(in_dim -> hidden) -> ReLU -> Dropout -> Linear(hidden -> num_classes).
    in_dim: in_dim of ADULT dataset is 104 after one-hot encoding and normalization.
    Layer names use fc1 and fc2 so the existing parameter-splitting utilities can extract classifier params.
    """

    def __init__(self, _model_id: int = None):
        super().__init__()
        self.model_id = _model_id
        # Use 104 features as in original file
        self.fc1 = nn.Linear(in_features=104, out_features=128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

    def get_model_type(self) -> str:
        return 'TabularAdultModel'

