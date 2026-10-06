"""Support functions for Oracle server configuration."""


def configure_oracle_servers(servers, unique_drift_ids, aggregator_factory):
    """
    Configure legacy or explicitly marked cumulative-concept Oracle servers.
    :param servers: Leaf servers sharing one Oracle routing mode.
    :param unique_drift_ids: Operation IDs used by legacy Oracle routing.
    :param aggregator_factory: Factory creating an independent Oracle strategy.
    :return: None.
    """
    signatures = [getattr(server, 'oracle_concept_signature', None) for server in servers]
    concept_mode = any(signature is not None for signature in signatures)
    if concept_mode:
        if any(signature is None for signature in signatures):
            raise ValueError('Oracle servers cannot mix legacy and cumulative-concept routing.')
        if any(not isinstance(signature, tuple) or not signature or
               any(type(label) is not int for label in signature) or
               set(signature) != set(range(len(signature))) for signature in signatures):
            raise ValueError('Oracle concept signatures must be tuples representing label permutations.')
        if len({len(signature) for signature in signatures}) != 1:
            raise ValueError('Oracle concept signatures must use the same class domain.')
        if len(set(signatures)) != len(signatures):
            raise ValueError('Oracle concept servers must have unique signatures.')
    elif len(servers) > len(unique_drift_ids):
        raise ValueError('Legacy Oracle requires an operation ID for every server.')

    strategies = [aggregator_factory() for _ in servers]
    for idx, (server, strategy) in enumerate(zip(servers, strategies)):
        # Cumulative concept identity is separate from the next swap operation ID.
        server.drift_id = None if concept_mode else unique_drift_ids[idx]
        server.strategy = strategy


def link_clients_to_oracle_concepts(clients, servers):
    """
    Route clients by committed mappings without changing swap operation IDs.
    :param clients: All clients with initialized cumulative mappings.
    :param servers: Stable flat concept-server list in server-ID order.
    :return: None.
    """
    by_mapping = {}
    for position, server in enumerate(servers):
        signature = getattr(server, 'oracle_concept_signature', None)
        if signature is None or signature in by_mapping or server.server_id != position:
            raise ValueError('Oracle concept servers need unique signatures and positional IDs.')
        by_mapping[signature] = server
    memberships = [[] for _ in servers]
    assignments = []
    seen = set()
    for client in clients:
        signature = getattr(client, 'oracle_concept_signature', None)
        if signature not in by_mapping or client.client_id in seen:
            raise ValueError('Every Oracle client needs a unique ID and a registered current mapping.')
        seen.add(client.client_id)
        server = by_mapping[signature]
        memberships[server.server_id].append(client.client_id)
        assignments.append((client, server.server_id))
    for server, members in zip(servers, memberships):
        server.client_ids = members
    for client, server_id in assignments:
        client.parent_server_id = server_id
