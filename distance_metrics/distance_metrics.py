"""
Description: This file consists the Averaging functions for FedAvg algorithm.

Author: Nisal Hemadasa
Date: 26-04-2025
Version: 1.0
"""

import torch
from collections import OrderedDict
from typing import List, Dict, Tuple, Any

import constants


class ModelDistanceHistory:
    """Structured model-distance records retained for diagnostics and future strategy optimization."""

    SCHEMA_VERSION = 1

    def __init__(self):
        self.metric = 'euclidean_l2'
        self.records = []

    def append(self, record: Dict[str, Any]) -> None:
        """
        Append one aggregation-event distance record.
        :param record: Structured distance record for one round.
        :return: None
        """
        self.records.append(record)

    def latest(self) -> Dict[str, Any] | None:
        """
        Return the latest record for consumers such as a future FedEx optimizer.
        :return: Latest record, or None when no distances have been collected.
        """
        return self.records[-1] if self.records else None

    def client_distances(self, round_idx: int | None = None) -> Dict[int, float]:
        """
        Return whole-model distances keyed by client ID for one record.
        :param round_idx: Round to retrieve; None selects the latest record.
        :return: Client ID to whole-model L2 distance mapping.
        """
        record = self.latest() if round_idx is None else next(
            (item for item in reversed(self.records) if item['round'] == round_idx), None)
        if record is None:
            return {}
        distances = {}
        for server_record in record['servers']:
            for model_record in server_record['models']:
                if model_record['model_id'] == 'primary':
                    distances.update({client_id: values['whole_model_l2']
                                      for client_id, values in model_record['clients'].items()})
        return distances

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a pickle-friendly representation of the history.
        :return: Versioned diagnostic structure.
        """
        return {'schema_version': self.SCHEMA_VERSION, 'metric': self.metric, 'records': self.records}

    def model_log(self) -> Dict[str, Any]:
        """
        Return the history without per-layer values for compact whole-model analysis.
        :return: Versioned whole-model distance log.
        """
        records = []
        for record in self.records:
            servers = []
            for server in record['servers']:
                models = []
                for model in server['models']:
                    clients = {
                        client_id: {key: value for key, value in values.items()
                                    if key not in ('layers', 'included_tensors')}
                        for client_id, values in model['clients'].items()
                    }
                    models.append({'model_id': model['model_id'], 'clients': clients})
                servers.append({key: value for key, value in server.items() if key != 'models'} |
                               {'models': models})
            records.append({'round': record['round'], 'phase': record['phase'], 'servers': servers})
        return {'schema_version': self.SCHEMA_VERSION, 'metric': self.metric, 'records': records}

    def layer_log(self) -> Dict[str, Any]:
        """
        Return per-layer distances with event and topology metadata.
        :return: Versioned layer-distance log.
        """
        records = []
        for record in self.records:
            servers = []
            for server in record['servers']:
                models = []
                for model in server['models']:
                    clients = {
                        client_id: {
                            'layers': values['layers'],
                            'included_tensors': values['included_tensors'],
                            'participated': values['participated'],
                            'last_local_update_round': values['last_local_update_round']
                        }
                        for client_id, values in model['clients'].items()
                    }
                    models.append({'model_id': model['model_id'], 'clients': clients})
                servers.append({key: value for key, value in server.items() if key != 'models'} |
                               {'models': models})
            records.append({'round': record['round'], 'phase': record['phase'], 'servers': servers})
        return {'schema_version': self.SCHEMA_VERSION, 'metric': self.metric, 'records': records}


def compute_model_l2_distance(server_model: OrderedDict, client_model: OrderedDict) -> Dict[str, Any]:
    """
    Compute true whole-model and per-tensor Euclidean distances between two state dictionaries.
    :param server_model: Reference server model state dictionary.
    :param client_model: Client model state dictionary.
    :return: Whole-model distance, per-tensor distances, and included tensor names.
    """
    server_keys = set(server_model.keys())
    client_keys = set(client_model.keys())
    if server_keys != client_keys:
        missing = sorted(server_keys - client_keys)
        extra = sorted(client_keys - server_keys)
        raise ValueError(f"Model state keys do not match; missing={missing}, extra={extra}")

    squared_total = 0.0
    layer_distances = {}
    for key, server_tensor in server_model.items():
        client_tensor = client_model[key]
        if server_tensor.shape != client_tensor.shape:
            raise ValueError(f"Model tensor shape mismatch for {key}: "
                             f"server={tuple(server_tensor.shape)}, client={tuple(client_tensor.shape)}")
        difference = client_tensor.detach().to(dtype=torch.float64, device='cpu') - \
            server_tensor.detach().to(dtype=torch.float64, device='cpu')
        squared_distance = torch.sum(difference * difference).item()
        layer_distances[key] = squared_distance ** 0.5
        squared_total += squared_distance

    return {
        'whole_model_l2': squared_total ** 0.5,
        'layers': layer_distances,
        'included_tensors': list(server_model.keys())
    }


def collect_model_distance_diagnostics(leaf_servers, clients, round_idx: int, sampled_client_ids: List[int],
                                       phase: str = 'post_aggregation_pre_distribution') -> Dict[str, Any]:
    """
    Compare each client with its assigned leaf server after aggregation and before distribution.
    :param leaf_servers: Leaf servers to which clients are assigned.
    :param clients: All client instances.
    :param round_idx: Aggregation round index.
    :param sampled_client_ids: Client IDs that contributed to this aggregation.
    :param phase: Event phase recorded with the diagnostic.
    :return: Structured record suitable for ModelDistanceHistory and future strategy use.
    """
    client_map = {client.client_id: client for client in clients}
    sampled_ids = set(sampled_client_ids)
    server_records = []

    for server in leaf_servers:
        model_pairs = []
        if server.model is not None:
            model_pairs.append(('primary', server.model, 'model'))
        elif server.multi_models is not None:
            model_pairs.extend((cluster_idx, model, 'fedrc_models')
                               for cluster_idx, model in enumerate(server.multi_models))

        model_records = []
        for model_id, server_model, client_attribute in model_pairs:
            client_records = {}
            for client_id in server.client_ids:
                client = client_map.get(client_id)
                if client is None or client.parent_server_id != server.server_id:
                    continue
                if client_attribute == 'model':
                    client_model = client.model
                else:
                    client_model = client.fedrc_models[model_id]
                if client_model is None:
                    continue
                values = compute_model_l2_distance(server_model.state_dict(), client_model.state_dict())
                values.update({
                    'participated': client_id in sampled_ids,
                    'last_local_update_round': 'initial_training' if round_idx == 0 else round_idx - 1
                })
                client_records[client_id] = values
            model_records.append({'model_id': model_id, 'clients': client_records})

        server_record = {
            'depth': 'leaf',
            'server_id': server.server_id,
            'server_abs_id': server.abs_id,
            'strategy': server.strategy.strategy_name,
            'models': model_records
        }
        if getattr(getattr(server, 'strategy', None), 'strategy_name', None) == \
                constants.RecoveryAlgorithm.FAIRFEDDRIFT:
            server_record['fairfeddrift_cluster_id'] = server.fairfeddrift_cluster_id
        server_records.append(server_record)

    return {'round': round_idx, 'phase': phase, 'servers': server_records}


def compute_euclidean_distance_weights(prev_edge_model: OrderedDict,
                                       client_model_params_list: List[OrderedDict]) -> Tuple[
                                        List[float], List[float], Dict[int, Dict[str, float]]]:
    """
    Computes normalized inverse-euclidean-distance weights for client models relative to the previous edge model.
    :param: prev_edge_model: State dict of the previous global model at the edge server
    :param: client_model_params_list: List of state dicts of the client models
    :return: weights: List of normalized inverse-euclidean-distance weights for each client model,
            client_model_distances: List of distances,
            client_layer_distances: Dictionary of distances of each separate layer from  the corresponding layer in the
            edge model for each client model
    """
    client_model_distances = []
    client_layer_distances = {}

    for index, client_params in enumerate(client_model_params_list):
        total_distance = 0.0
        layer_distance = {}

        for key in prev_edge_model.keys():
            if key in client_params and prev_edge_model[key].shape == client_params[key].shape:
                diff = client_params[key] - prev_edge_model[key]

                # Compute the L2 (Euclidean) distance for this layer
                l2_distance = diff.norm(p=2)
                total_distance += l2_distance.item()
                layer_distance[key] = l2_distance.item()

        client_model_distances.append(total_distance)
        client_layer_distances[index] = layer_distance

    distances = torch.tensor(client_model_distances)

    # Normalize using direct distances, as per HAF-Edge Equation (4)
    if torch.sum(distances) == 0:
        # Edge case: if all distances are zero, assign equal weights
        weights = torch.ones_like(distances) / len(distances)
    else:
        weights = distances / distances.sum()

    return weights.tolist(), client_model_distances, client_layer_distances

