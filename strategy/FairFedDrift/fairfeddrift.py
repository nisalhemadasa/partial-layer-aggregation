"""
Description: This module defines the FairFedDrift strategy entry point.

Scalar-loss selection, cluster creation, merging and weighted upload aggregation are implemented.
"""
import math
from copy import deepcopy
from typing import Dict, OrderedDict

import torch
from torch import nn

import constants
from models.utils import set_parameters
from .utils import (average_fairfeddrift_parameters, calculate_fairfeddrift_merge_distances,
                    update_fairfeddrift_complete_linkage)


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
    def __init__(self, strategy_name: str, parameters: Dict = None):
        """
        Initialize the strategy identifier and independent client loss references.
        :param strategy_name: Recovery algorithm name.
        :param parameters: Resolved FairFedDrift experiment configuration, if provided.
        """
        self.strategy_name = strategy_name
        self.parameters = (resolve_fairfeddrift_parameters(parameters) if parameters is not None else None)
        self.previous_losses = {}
        self.cluster_models = {}
        self.client_assignments = {}
        self.assignment_history = {}
        self.client_data_histories = {}
        self.last_timestep_key = None
        self.current_timestep_start_round = None
        self.runtime_history = []
        self._next_cluster_id = 0

    def record_assignment(self, client_id: int, arrival_round: int) -> None:
        """
        Record the current learned assignment for a newly retained data arrival.
        :param client_id: Previously assigned client.
        :param arrival_round: Nonnegative arrival round; duplicate recording is idempotent.
        :return: None. History expiry is handled by later runtime history wiring.
        """
        if client_id not in self.client_assignments:
            raise ValueError("Assign the client before recording its data arrival.")
        if isinstance(arrival_round, bool) or not isinstance(arrival_round, int) or arrival_round < 0:
            raise ValueError("Assignment arrival round must be a non-negative integer.")
        history = self.assignment_history.setdefault(client_id, {})
        history.setdefault(arrival_round, self.client_assignments[client_id])

    def merge_clusters(self, histories, timestep_start_round: int, loss_threshold: float,
                       batch_size: int = 64) -> list[dict]:
        """
        Merge eligible clusters in increasing distance order, updating learned assignments.
        Stage all changes before committing, so evaluation/copy/load failures leave
        strategy state intact. Equal distances follow active cluster order.
        :param histories: ClientDataHistory objects advanced to the current round.
        :param timestep_start_round: First round of current timestep, excluded from comparisons.
        :param loss_threshold: Shared scalar loss-increase threshold.
        :param batch_size: Batch size for historical loss evaluation.
        :return: Scalar-only records of source IDs, new ID, distance and historical counts.
        """
        if not self.cluster_models:
            raise ValueError("Initialize FairFedDrift clusters before merging.")
        if any(cluster_id not in self.cluster_models for cluster_id in self.client_assignments.values()):
            raise ValueError("Current assignment references an inactive cluster.")
        if any(arrivals and client not in histories for client, arrivals in self.assignment_history.items()):
            raise ValueError("Supply every client history when merging clusters.")
        # Keep only assignment references backed by retained snapshots, including
        # current-timestep arrivals (excluded from loss/counts but relabeled on merge).
        retained_assignments = {}
        for client_id, history in histories.items():
            retained_assignments[client_id] = {}
            for arrival, _ in history.records():
                cluster_id = self.assignment_history.get(client_id, {}).get(arrival)
                if cluster_id not in self.cluster_models:
                    raise ValueError("Retained arrival requires an active historical assignment.")
                retained_assignments[client_id][arrival] = cluster_id
        distances, counts = calculate_fairfeddrift_merge_distances(
            self.cluster_models, histories, retained_assignments, timestep_start_round,
            loss_threshold, batch_size)
        models = dict(self.cluster_models)
        assignments = dict(self.client_assignments)
        next_id = self._next_cluster_id
        records = []
        while True:
            ids = list(models)
            pairs = [(distances[i][j], i, j) for pos, i in enumerate(ids)
                     for j in ids[pos + 1:] if math.isfinite(distances[i][j])]
            if not pairs:
                break
            distance, first, second = min(pairs, key=lambda pair: pair[0])
            parameters = average_fairfeddrift_parameters(
                [models[first].state_dict(), models[second].state_dict()],
                [counts[first], counts[second]])
            merged_model = deepcopy(models[first])
            set_parameters(merged_model, parameters)
            distances = update_fairfeddrift_complete_linkage(distances, first, second, next_id)
            records.append({'source_ids': (first, second), 'cluster_id': next_id,
                            'distance': distance, 'sample_counts': (counts[first], counts[second]),
                            'sample_count': counts[first] + counts[second]})
            counts[next_id] = counts.pop(first) + counts.pop(second)
            del models[first]
            del models[second]
            models[next_id] = merged_model
            assignments = {client: next_id if cluster in (first, second) else cluster
                           for client, cluster in assignments.items()}
            retained_assignments = {
                client: {arrival: next_id if cluster in (first, second) else cluster
                         for arrival, cluster in arrivals.items()}
                for client, arrivals in retained_assignments.items()}
            next_id += 1
        self.cluster_models = models
        self.client_assignments = assignments
        self.assignment_history = retained_assignments
        self._next_cluster_id = next_id
        return records

    def initialize_clusters(self, initial_model: nn.Module) -> int:
        """
        Initialize the model registry once from the framework's existing model.
        :param initial_model: Initial server model, already on the configured device.
        :return: Initial cluster ID; the registry owns an independent model copy.
        """
        if self._next_cluster_id != 0:
            raise ValueError("FairFedDrift clusters are already initialized.")
        if not isinstance(initial_model, nn.Module):
            raise ValueError("FairFedDrift initialization requires a PyTorch model.")
        self.cluster_models[0] = deepcopy(initial_model)
        self._next_cluster_id = 1
        return 0

    def assign_client(self, client_id: int, candidate_losses: Dict[int, float],
                      loss_threshold: float) -> tuple[int, bool]:
        """
        Assign a client to an acceptable model or create an independent cluster.
        New clusters copy the first currently active model, not the best failing
        candidate. IDs increase monotonically and are not server-list positions.
        :param client_id: Nonnegative client ID, unrelated to ground-truth drift IDs.
        :param candidate_losses: Losses for active candidates frozen for this decision event.
        :param loss_threshold: Shared absolute loss-increase threshold.
        :return: Assigned cluster ID and whether a new cluster was created.
        """
        if not self.cluster_models:
            raise ValueError("FairFedDrift clusters must be initialized before assignment.")
        if isinstance(client_id, bool) or not isinstance(client_id, int) or client_id < 0:
            raise ValueError("FairFedDrift client IDs must be non-negative integers.")
        if not isinstance(candidate_losses, dict) or not candidate_losses:
            raise ValueError("FairFedDrift assignment requires candidate losses.")
        if any(cluster_id not in self.cluster_models for cluster_id in candidate_losses):
            raise ValueError("FairFedDrift candidate losses reference an inactive cluster.")
        previous_loss = self.previous_losses.get(client_id)
        selected_id, _ = self.select_model(client_id, candidate_losses, loss_threshold)
        created = selected_id is None
        if created:
            try:
                new_model = deepcopy(next(iter(self.cluster_models.values())))
            except Exception:
                # Failed copies must not commit a decision or consume a cluster ID.
                if previous_loss is None:
                    self.previous_losses.pop(client_id, None)
                else:
                    self.previous_losses[client_id] = previous_loss
                raise
            selected_id = self._next_cluster_id
            self.cluster_models[selected_id] = new_model
            self._next_cluster_id += 1
        self.client_assignments[client_id] = selected_id
        return selected_id, created

    def select_model(self, client_id: int, candidate_losses: Dict[int, float],
                     loss_threshold: float) -> tuple[int | None, float]:
        """
        Select the lowest-loss acceptable candidate and update this client's reference.
        The single-group adaptation uses the scalar rule documented in the
        integration plan (upstream FairFedDrift's fed_drift.py). Ties retain the
        first candidate in active-model order. A new client starts at loss 1000.
        :param client_id: Client whose independent previous loss is compared.
        :param candidate_losses: Nonempty cluster-ID to whole-client loss mapping in active order.
        :param loss_threshold: Shared finite, nonnegative absolute loss-increase threshold.
        :return: Selected cluster ID, or None requesting creation, and next reference loss.
        """
        threshold = resolve_fairfeddrift_parameters(
            {'fairfeddrift_loss_threshold': loss_threshold})['fairfeddrift_loss_threshold']
        if not isinstance(candidate_losses, dict) or not candidate_losses:
            raise ValueError("FairFedDrift selection requires candidate losses.")
        for cluster_id, loss in candidate_losses.items():
            if isinstance(cluster_id, bool) or not isinstance(cluster_id, int) or cluster_id < 0:
                raise ValueError("FairFedDrift cluster IDs must be non-negative integers.")
            if (isinstance(loss, bool) or not isinstance(loss, (int, float)) or
                    not math.isfinite(loss) or loss < 0):
                raise ValueError("FairFedDrift candidate losses must be finite non-negative numbers.")
        best_id = min(candidate_losses, key=candidate_losses.get)
        best_loss = float(candidate_losses[best_id])
        previous_loss = self.previous_losses.get(client_id, 1000.0)
        selected_id = best_id if best_loss <= previous_loss + threshold else None
        # Also retain the best failing loss when a new cluster is required.
        self.previous_losses[client_id] = best_loss
        return selected_id, best_loss

    def aggregate_models(self, server_model: nn.Module,
                         client_model_params_dict: Dict[int, OrderedDict],
                         client_sample_counts_dict: Dict[int, int],
                         history_model_params_dict: Dict[int, OrderedDict] = None,
                         history_sample_counts_dict: Dict[int, int] = None) -> None:
        """
        Aggregate current and retained-history uploads by their sample counts.
        :param server_model: Cluster server model updated by the aggregate.
        :param client_model_params_dict: Current-assignment state dictionaries keyed by client ID.
        :param client_sample_counts_dict: Current-upload sample counts keyed by client ID.
        :param history_model_params_dict: Optional historical state dictionaries keyed by client ID.
        :param history_sample_counts_dict: Optional historical sample counts keyed by client ID.
        :return: None.
        """
        if not isinstance(server_model, nn.Module):
            raise ValueError("FairFedDrift aggregation requires a PyTorch server model.")
        upload_groups = (
            (client_model_params_dict, client_sample_counts_dict, 'current'),
            ({} if history_model_params_dict is None else history_model_params_dict,
             {} if history_sample_counts_dict is None else history_sample_counts_dict, 'history')
        )
        server_parameters = server_model.state_dict()
        expected_keys = list(server_parameters)
        parameter_sets, sample_counts = [], []
        for parameters_by_client, counts_by_client, source in upload_groups:
            if not isinstance(parameters_by_client, dict) or not isinstance(counts_by_client, dict):
                raise ValueError("FairFedDrift " + source + " uploads and counts must be dictionaries.")
            if set(parameters_by_client) != set(counts_by_client):
                raise ValueError("FairFedDrift " + source + " uploads and sample counts must have identical client IDs.")
            for client_id, parameters in parameters_by_client.items():
                if isinstance(client_id, bool) or not isinstance(client_id, int) or client_id < 0:
                    raise ValueError("FairFedDrift client IDs must be non-negative integers.")
                if not isinstance(parameters, dict) or list(parameters) != expected_keys:
                    raise ValueError("FairFedDrift " + source + " model keys must match the server model for client "
                                     + str(client_id) + ".")
                for key, server_tensor in server_parameters.items():
                    tensor = parameters[key]
                    if (not isinstance(tensor, torch.Tensor) or tensor.shape != server_tensor.shape or
                            tensor.dtype != server_tensor.dtype):
                        raise ValueError("FairFedDrift tensor shape or dtype mismatch for " + source + " client "
                                         + str(client_id) + " and key " + key + ".")
                parameter_sets.append(parameters)
                sample_counts.append(counts_by_client[client_id])
        if not parameter_sets:
            raise ValueError("FairFedDrift aggregation requires at least one current or historical upload.")
        aggregated_parameters = average_fairfeddrift_parameters(parameter_sets, sample_counts)
        set_parameters(server_model, aggregated_parameters)


def aggregator_fn(parameters: Dict = None):
    """
    Return a FairFedDrift aggregation strategy.
    :param parameters: Resolved FairFedDrift experiment configuration, if provided.
    :return: FairFedDrift strategy instance.
    """
    return FairFedDrift(strategy_name=constants.RecoveryAlgorithm.FAIRFEDDRIFT, parameters=parameters)
