"""Oracle-specific observation of the unchanged whole-class swap implementation."""
import copy

from drift_concepts.drift import Drift
from drift_concepts.utils import compose_label_mapping
from strategy.Oracle.utils import link_clients_to_oracle_concepts


class OracleTrackedDrift(Drift):
    def __init__(self, source, clients, servers):
        """
        Copy configured drift state without recomputing groups or random choices.
        :param source: Fresh ordinary Drift instance to continue tracking.
        :param clients: All run clients, retained by reference.
        :param servers: Stable cumulative-concept servers, retained by reference.
        """
        self.__dict__.update(copy.deepcopy(source.__dict__))
        self._oracle_clients = tuple(clients)
        self._oracle_servers = servers
        self.oracle_mapping_history = []
        self._oracle_failed = False

    def swap_labels(self, clients, class_pair_to_swap, verbose=False):
        """
        Commit concept state only after the existing dataset swap succeeds.
        :param clients: Actual client subset passed by the existing drift dispatcher.
        :param class_pair_to_swap: Ordered pairs actually applied to this subset.
        :param verbose: Forwarded to the existing swap implementation.
        :return: Original swap result.
        """
        if self._oracle_failed:
            raise RuntimeError('Oracle drift failed previously; start a fresh experiment.')
        pairs = tuple(tuple(pair) for pair in class_pair_to_swap)
        supplied = {client.client_id: client for client in clients}
        known = {client.client_id: client for client in self._oracle_clients}
        affected = list(self.drifted_client_indices or [])
        registered = {server.oracle_concept_signature for server in self._oracle_servers}
        pending = []
        if len(supplied) != len(clients) or len(set(affected)) != len(affected):
            raise ValueError('Oracle swap requires unique affected clients.')
        for client_id in affected:
            if client_id not in supplied or supplied[client_id] is not known.get(client_id):
                raise ValueError('Oracle swap received an unknown client instance.')
            client = supplied[client_id]
            before = client.oracle_concept_signature
            after = compose_label_mapping(before, pairs)
            if after not in registered:
                raise ValueError('Applied swaps reach an unregistered Oracle concept.')
            pending.append((client, before, after))
        try:
            result = super().swap_labels(clients, pairs, verbose)
            for client, before, after in pending:
                client.oracle_concept_signature = after
            link_clients_to_oracle_concepts(self._oracle_clients, self._oracle_servers)
        except Exception:
            # Dataset mutations may be partial; never attempt to continue this run.
            self._oracle_failed = True
            raise
        for client, before, after in pending:
            self.oracle_mapping_history.append({
                'round': self.current_round, 'step': self.current_drift_step,
                'client_id': client.client_id, 'operation_id': client.drift_id,
                'swaps': pairs, 'before_mapping': before, 'after_mapping': after,
                'server_id': client.parent_server_id,
            })
        return result
