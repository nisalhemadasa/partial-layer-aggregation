"""
Description: This module defines the FairFedDrift strategy entry point.

Client-loss-based clustering and aggregation for the PaLA adaptation are pending implementation.
"""
import math
from typing import Dict, OrderedDict

from torch import nn

import constants


FAIRFEDDRIFT_DEFAULTS = {
    'fairfeddrift_window': 100,  # Communication rounds, not data timesteps.
    'fairfeddrift_rounds_per_timestep': 1,
    'fairfeddrift_seed': 42,
}


def resolve_fairfeddrift_parameters(parameters: Dict) -> Dict:
    """
    Resolve FairFedDrift settings without modifying the input configuration.
    :param parameters: Recovery settings including an explicit local-loss threshold.
    :return: Validated settings; window counts communication rounds, None means unbounded.
    """
    if not isinstance(parameters, dict):
        raise ValueError("FairFedDrift parameters must be a dictionary.")
    obsolete_keys = [key for key in ('fairfeddrift_threshold_privileged',
                                    'fairfeddrift_threshold_unprivileged') if key in parameters]
    if obsolete_keys:
        raise ValueError("Obsolete FairFedDrift settings: " + ', '.join(obsolete_keys) +
                         ". Remove these keys and set fairfeddrift_loss_threshold for the single-loss adaptation.")
    resolved = dict(FAIRFEDDRIFT_DEFAULTS)
    resolved.update({key: parameters[key] for key in resolved if key in parameters})
    threshold = parameters.get('fairfeddrift_loss_threshold')
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or
            not math.isfinite(threshold) or threshold < 0):
        raise ValueError("fairfeddrift_loss_threshold is required and must be a finite non-negative number.")
    resolved['fairfeddrift_loss_threshold'] = float(threshold)

    window = resolved['fairfeddrift_window']
    if window is not None and (isinstance(window, bool) or not isinstance(window, int) or window <= 0):
        raise ValueError("fairfeddrift_window must be None (unbounded) or a positive integer of communication rounds.")
    rounds = resolved['fairfeddrift_rounds_per_timestep']
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds <= 0:
        raise ValueError("fairfeddrift_rounds_per_timestep must be a positive integer.")
    seed = resolved['fairfeddrift_seed']
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("fairfeddrift_seed must be an integer.")
    return resolved


def validate_fairfeddrift_setup(server_tree_layout, client_select_fraction, resolved_device) -> None:
    """
    Validate the initial supported execution setup without changing devices.
    :param server_tree_layout: Number of servers per hierarchy level.
    :param client_select_fraction: Fraction of clients participating in each round.
    :param resolved_device: Device returned by configure_device() or get_device().
    :return: None.
    """
    if (not isinstance(server_tree_layout, (list, tuple)) or len(server_tree_layout) != 1 or
            isinstance(server_tree_layout[0], bool) or not isinstance(server_tree_layout[0], int) or
            server_tree_layout[0] <= 0):
        raise ValueError("FairFedDrift requires a single-level server layout with a positive server count.")
    if (isinstance(client_select_fraction, bool) or not isinstance(client_select_fraction, (int, float)) or
            client_select_fraction != 1):
        raise ValueError("FairFedDrift currently requires full participation (client_select_fraction=1).")
    if getattr(resolved_device, 'type', None) != 'cpu':
        raise ValueError("FairFedDrift currently requires a resolved CPU device.")


class FairFedDrift:
    def __init__(self, strategy_name: str):
        """
        Initialize the FairFedDrift strategy identifier.
        :param strategy_name: Recovery algorithm name.
        """
        self.strategy_name = strategy_name

    def aggregate_models(self, server_model: nn.Module,
                         client_model_params_dict: Dict[int, OrderedDict]) -> None:
        """
        Reserve the aggregation entry point for FairFedDrift implementation.
        :param server_model: Server model to receive aggregated parameters.
        :param client_model_params_dict: Client state dictionaries keyed by client ID.
        :return: None.
        """
        raise NotImplementedError("FairFedDrift aggregation is not implemented yet.")


def aggregator_fn():
    """
    Return a FairFedDrift aggregation strategy.
    :return: FairFedDrift strategy instance.
    """
    return FairFedDrift(strategy_name=constants.RecoveryAlgorithm.FAIRFEDDRIFT)
