"""Tests for learned FairFedDrift cluster-to-server routing."""
from types import SimpleNamespace
import unittest

from federated_network.utils import link_clients_to_fairfeddrift_servers


class FairFedDriftLinkingTests(unittest.TestCase):
    def setUp(self):
        self.servers = [SimpleNamespace(client_ids=[99]), SimpleNamespace(client_ids=[98])]
        self.clients = [SimpleNamespace(client_id=10, parent_server_id=None),
                        SimpleNamespace(client_id=11, parent_server_id=None),
                        SimpleNamespace(client_id=12, parent_server_id=None)]

    def test_learned_ids_map_to_positions_without_becoming_server_indices(self):
        mapping = link_clients_to_fairfeddrift_servers(
            self.servers, self.clients, [7, 12], {10: 12, 11: 7, 12: 12})
        self.assertEqual(mapping, {7: 0, 12: 1})
        self.assertEqual([client.parent_server_id for client in self.clients], [1, 0, 1])
        self.assertEqual([server.client_ids for server in self.servers], [[11], [10, 12]])

    def test_relinking_replaces_old_client_memberships(self):
        link_clients_to_fairfeddrift_servers(
            self.servers, self.clients, [7, 12], {10: 7, 11: 12, 12: 7})
        self.assertEqual([server.client_ids for server in self.servers], [[10, 12], [11]])

    def test_rejects_mismatched_server_count_and_incomplete_assignments(self):
        with self.assertRaises(ValueError):
            link_clients_to_fairfeddrift_servers(self.servers[:1], self.clients, [7, 12], {})
        with self.assertRaises(ValueError):
            link_clients_to_fairfeddrift_servers(self.servers, self.clients, [7, 12], {10: 7})

    def test_rejects_inactive_cluster_and_duplicate_ids(self):
        with self.assertRaises(ValueError):
            link_clients_to_fairfeddrift_servers(
                self.servers, self.clients, [7, 12], {10: 99, 11: 7, 12: 12})
        with self.assertRaises(ValueError):
            link_clients_to_fairfeddrift_servers(
                self.servers, self.clients, [7, 7], {10: 7, 11: 7, 12: 7})


if __name__ == '__main__':
    unittest.main()
