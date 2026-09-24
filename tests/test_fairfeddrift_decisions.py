"""Tests for the frozen-candidate FairFedDrift decision pass."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from federated_network.utils import run_fairfeddrift_decisions
from strategy.FairFedDrift import aggregator_fn


class FairFedDriftDecisionPassTests(unittest.TestCase):
    def make_server(self, strategy):
        return SimpleNamespace(server_id=0, abs_id=0, strategy=strategy, model=nn.Linear(1, 2),
                               client_ids=[], child_server_ids=[], parent_server_id=None,
                               drift_id=None, multi_models=None, fedex_alpha=1.0)

    def make_client(self, client_id):
        dataset = TensorDataset(torch.tensor([[float(client_id)]]), torch.tensor([0]))
        return SimpleNamespace(client_id=client_id, parent_server_id=0,
                               trainloader=DataLoader(dataset, batch_size=1, shuffle=True))

    def test_new_clusters_are_not_candidates_midway_through_batch(self):
        strategy = aggregator_fn({'fairfeddrift_loss_threshold': 0.1})
        servers = [self.make_server(strategy)]
        clients = [self.make_client(10), self.make_client(11)]
        strategy.previous_losses.update({10: 0.0, 11: 1000.0})

        def loss_for_client(_model, loader):
            client_id = int(loader.dataset[0][0].item())
            return 0.5 if client_id == 10 else 0.2

        with patch('federated_network.utils.evaluate_fairfeddrift_loss', side_effect=loss_for_client):
            records = run_fairfeddrift_decisions(servers, clients)

        self.assertEqual([record['candidate_cluster_ids'] for record in records], [(0,), (0,)])
        self.assertTrue(records[0]['created_cluster'])
        self.assertEqual(records[0]['assigned_cluster_id'], 1)
        self.assertFalse(records[1]['created_cluster'])
        self.assertEqual(records[1]['assigned_cluster_id'], 0)
        self.assertEqual([client.parent_server_id for client in clients], [1, 0])
        self.assertEqual([server.client_ids for server in servers], [[11], [10]])
        self.assertIs(servers[1].model, strategy.cluster_models[1])

    def test_all_candidates_use_one_prepared_loader_per_client(self):
        strategy = aggregator_fn({'fairfeddrift_loss_threshold': 0.5})
        servers = [self.make_server(strategy)]
        clients = [self.make_client(10), self.make_client(11)]
        strategy.initialize_clusters(servers[0].model)
        strategy.cluster_models[7] = deepcopy(strategy.cluster_models[0])
        strategy._next_cluster_id = 8
        loader_ids_by_client = {10: set(), 11: set()}

        def record_loader(_model, loader):
            client_id = int(loader.dataset[0][0].item())
            loader_ids_by_client[client_id].add(id(loader))
            return 0.2

        with patch('federated_network.utils.evaluate_fairfeddrift_loss', side_effect=record_loader) as evaluate:
            records = run_fairfeddrift_decisions(servers, clients)

        self.assertEqual(evaluate.call_count, 4)
        self.assertEqual(len(loader_ids_by_client[10]), 1)
        self.assertEqual(len(loader_ids_by_client[11]), 1)
        self.assertEqual([record['candidate_cluster_ids'] for record in records], [(0, 7), (0, 7)])
        self.assertEqual([server.fairfeddrift_cluster_id for server in servers], [0, 7])
        self.assertEqual([client.parent_server_id for client in clients], [0, 0])


if __name__ == '__main__':
    unittest.main()
