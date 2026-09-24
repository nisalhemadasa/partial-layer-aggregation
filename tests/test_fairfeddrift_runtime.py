"""Tests for FairFedDrift runtime state ownership."""
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from federated_network.utils import initialize_fairfeddrift_runtime
from strategy.FairFedDrift import aggregator_fn
from strategy.FedAvg import aggregator_fn as fedavg_aggregator_fn


class FairFedDriftRuntimeTests(unittest.TestCase):
    def make_server(self, strategy):
        return SimpleNamespace(strategy=strategy, model=nn.Linear(1, 1), client_ids=[], abs_id=0)

    def test_initial_cluster_copies_post_warmup_model_and_becomes_live_server_model(self):
        strategy = aggregator_fn()
        server = self.make_server(strategy)
        with torch.no_grad():
            server.model.weight.fill_(4.25)
            server.model.bias.fill_(-2.0)

        shared = initialize_fairfeddrift_runtime([server])

        self.assertIs(shared, strategy)
        self.assertIs(server.model, shared.cluster_models[0])
        self.assertAlmostEqual(server.model.weight.item(), 4.25)
        self.assertAlmostEqual(server.model.bias.item(), -2.0)

    def test_flat_servers_share_one_strategy_and_initial_model_registry(self):
        first_strategy = aggregator_fn()
        second_strategy = aggregator_fn()
        servers = [self.make_server(first_strategy), self.make_server(second_strategy)]
        with torch.no_grad():
            servers[0].model.weight.fill_(3.0)

        shared = initialize_fairfeddrift_runtime(servers)

        self.assertIs(servers[1].strategy, shared)
        self.assertIs(servers[0].model, shared.cluster_models[0])
        self.assertAlmostEqual(shared.cluster_models[0].weight.item(), 3.0)

    def test_repeated_initialization_does_not_reset_live_cluster_weights(self):
        server = self.make_server(aggregator_fn())
        shared = initialize_fairfeddrift_runtime([server])
        with torch.no_grad():
            server.model.weight.fill_(8.0)

        self.assertIs(initialize_fairfeddrift_runtime([server]), shared)
        self.assertIs(server.model, shared.cluster_models[0])
        self.assertAlmostEqual(shared.cluster_models[0].weight.item(), 8.0)

    def test_rejects_mixed_recovery_methods(self):
        servers = [self.make_server(aggregator_fn()),
                   self.make_server(fedavg_aggregator_fn())]
        with self.assertRaises(ValueError):
            initialize_fairfeddrift_runtime(servers)


if __name__ == '__main__':
    unittest.main()
