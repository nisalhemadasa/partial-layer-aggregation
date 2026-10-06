"""Shared experiment seeding used by the existing entry point and network."""
import random

import numpy as np
import torch


def configure_random_seed(seed):
    """
    Seed Python, NumPy and PyTorch without changing the experiment device.
    :param seed: Integer in NumPy's supported unsigned 32-bit seed range.
    :return: None.
    """
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError('Experiment seed must be an integer in [0, 2**32).')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
