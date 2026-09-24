"""Integration checks for cluster-tagged, sample-weighted server uploads."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import TensorDataset

import constants
from federated_network.server import Server, model_aggregation
from strategy.FairFedDrift import aggregator_fn


class FairFedDriftServerAggregationTests(unittest.TestCase):
    def make_server(self, cluster_id, server_id):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        server = Server(server_id, server_id, aggregator_fn({'fairfeddrift_loss_threshold': 1.0}),
                        model, 1, 0.9)
        server.fairfeddrift_cluster_id = cluster_id
        return server

    def make_client(self, client_id, weight, local_sample_count, training_cluster_id=None,
                    parent_server_id=None):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(weight)
        dataset = TensorDataset(torch.zeros(local_sample_count, 1), torch.zeros(local_sample_count, dtype=torch.long))
        return SimpleNamespace(client_id=client_id, model=model, local_trainset=dataset,
                               fairfeddrift_training_cluster_id=training_cluster_id,
                               parent_server_id=parent_server_id,
                               fairfeddrift_history_uploads={}, fairfeddrift_history_sample_counts={})

    def aggregate(self, servers, clients):
        model_aggregation([servers], None, clients, drift=None, ema_weight=0.9,
                          _is_server_has_test_data=False)

    def test_untagged_initial_warmup_uploads_aggregate_into_cluster_zero(self):
        server = self.make_server(cluster_id=0, server_id=0)
        clients = [self.make_client(1, 2.0, 1), self.make_client(2, 6.0, 3)]

        self.aggregate([server], clients)

        self.assertAlmostEqual(server.model.weight.item(), 5.0)

    def test_training_cluster_tags_override_later_parent_server_positions(self):
        servers = [self.make_server(cluster_id=0, server_id=0),
                   self.make_server(cluster_id=7, server_id=1)]
        clients = [self.make_client(4, 10.0, 1, training_cluster_id=7, parent_server_id=0),
                   self.make_client(9, 2.0, 4, training_cluster_id=0, parent_server_id=1)]

        self.aggregate(servers, clients)

        self.assertAlmostEqual(servers[0].model.weight.item(), 2.0)
        self.assertAlmostEqual(servers[1].model.weight.item(), 10.0)

    def test_current_and_history_uploads_route_to_each_clusters_server(self):
        servers = [self.make_server(cluster_id=0, server_id=0),
                   self.make_server(cluster_id=7, server_id=1)]
        clients = [self.make_client(4, 10.0, 1, training_cluster_id=7, parent_server_id=0),
                   self.make_client(9, 2.0, 4, training_cluster_id=0, parent_server_id=1)]
        clients[0].fairfeddrift_history_uploads = {
            0: {'weight': torch.tensor([[4.0]])},
            7: {'weight': torch.tensor([[8.0]])}}
        clients[0].fairfeddrift_history_sample_counts = {0: 2, 7: 3}
        clients[1].fairfeddrift_history_uploads = {0: {'weight': torch.tensor([[6.0]])}}
        clients[1].fairfeddrift_history_sample_counts = {0: 2}

        self.aggregate(servers, clients)

        self.assertAlmostEqual(servers[0].model.weight.item(), 3.5)
        self.assertAlmostEqual(servers[1].model.weight.item(), 8.5)

    def test_history_uploads_referencing_inactive_clusters_fail_before_mutation(self):
        server = self.make_server(cluster_id=0, server_id=0)
        client = self.make_client(4, 10.0, 1, training_cluster_id=0, parent_server_id=0)
        client.fairfeddrift_history_uploads = {9: {'weight': torch.tensor([[3.0]])}}
        client.fairfeddrift_history_sample_counts = {9: 1}
        with self.assertRaisesRegex(ValueError, 'inactive cluster'):
            self.aggregate([server], [client])
        self.assertAlmostEqual(server.model.weight.item(), 0.0)

    def test_inactive_or_mixed_upload_tags_fail_before_mutating_server_models(self):
        servers = [self.make_server(cluster_id=0, server_id=0),
                   self.make_server(cluster_id=7, server_id=1)]
        clients = [self.make_client(4, 10.0, 1, training_cluster_id=99),
                   self.make_client(9, 2.0, 4, training_cluster_id=0)]
        original = [server.model.weight.detach().clone() for server in servers]

        with self.assertRaises(ValueError):
            self.aggregate(servers, clients)

        for server, weight in zip(servers, original):
            self.assertTrue(torch.equal(server.model.weight.detach(), weight))

    def test_server_train_dispatches_to_fairfeddrift_weighted_aggregation(self):
        server = self.make_server(cluster_id=0, server_id=0)
        uploads = {1: {'weight': torch.tensor([[1.0]])},
                   2: {'weight': torch.tensor([[5.0]])}}
        server.train(uploads, client_sample_counts={1: 1, 2: 3})

        self.assertAlmostEqual(server.model.weight.item(), 4.0)
        self.assertEqual(server.strategy.strategy_name, constants.RecoveryAlgorithm.FAIRFEDDRIFT)

    def test_server_dispatches_historical_uploads_and_counts(self):
        server = self.make_server(cluster_id=7, server_id=0)
        server.train({1: {'weight': torch.tensor([[1.0]])}}, client_sample_counts={1: 1},
                     fairfeddrift_history_parameters={2: {'weight': torch.tensor([[5.0]])}},
                     fairfeddrift_history_sample_counts={2: 3})
        self.assertAlmostEqual(server.model.weight.item(), 4.0)


if __name__ == '__main__':
    unittest.main()
