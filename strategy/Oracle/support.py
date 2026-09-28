"""Explicit preparation of a cumulative-concept Oracle experiment."""
import copy

import constants
from drift_concepts.utils import reachable_label_mappings
from strategy.Oracle.utils import configure_oracle_servers, link_clients_to_oracle_concepts


def prepare_oracle_cumulative_routing(network, num_classes):
    """
    Prepare a fresh flat Oracle network before its existing simulation call.
    :param network: Constructed network using Oracle as base and recovery strategy.
    :param num_classes: Complete dataset class count, not a client's observed count.
    :return: Oracle tracking drift instance, including its in-memory mapping history.
    """
    from drift_concepts.drift import Drift
    from drift_concepts.oracle_drift import OracleTrackedDrift
    from strategy.Oracle import aggregator_fn

    parameters = network.drift_recovery_parameters
    if (parameters.get('recovery_method') != constants.RecoveryAlgorithm.ORACLE or
            parameters.get('base_aggregation_method') != constants.RecoveryAlgorithm.ORACLE or
            network.initial_aggregation_method != constants.RecoveryAlgorithm.ORACLE):
        raise ValueError('Cumulative Oracle requires Oracle throughout the experiment.')
    if len(network.server_hierarchy) != 1 or not network.server_hierarchy[0]:
        raise ValueError('Cumulative Oracle requires a nonempty flat server layout.')
    source = network.drift
    if (type(source) is not Drift or source.is_drift or source.is_drift_end or
            source.is_already_applied or source.current_drift_step != -1 or source.current_round != 0 or
            any(hasattr(client, 'oracle_concept_signature') for client in network.clients)):
        raise ValueError('Prepare cumulative Oracle once, before the experiment starts.')
    if (source.is_synchronous or
            source.drift_mode != constants.DriftMode.LABEL_SWAP_INCREMENTAL_STEPS or
            any(value != 1 for value in source.label_swap_percentage_steps)):
        raise ValueError('Cumulative Oracle currently supports asynchronous whole-class incremental swaps only.')
    client_ids = [client.client_id for client in network.clients]
    if client_ids != list(range(len(client_ids))):
        raise ValueError('Existing Oracle aggregation requires positional client IDs.')
    signatures = reachable_label_mappings(source, client_ids, num_classes)
    template = network.server_hierarchy[0][0]
    servers = []
    for index, signature in enumerate(signatures):
        server = copy.deepcopy(template)
        server.server_id = index
        server.abs_id = index
        server.parent_server_id = None
        server.child_server_ids = []
        server.client_ids = []
        server.oracle_concept_signature = signature
        servers.append(server)
    configure_oracle_servers(servers, source.unique_drift_ids, aggregator_fn)
    tracked = OracleTrackedDrift(source, network.clients, servers)
    for client in network.clients:
        client.oracle_concept_signature = tuple(range(num_classes))
    link_clients_to_oracle_concepts(network.clients, servers)
    network.server_hierarchy[0][:] = servers
    network.drift = tracked
    return tracked
