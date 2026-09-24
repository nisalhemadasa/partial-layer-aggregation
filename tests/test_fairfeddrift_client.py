"""Checks ordinary FairFedDrift client downloads and local fit behavior."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import constants
from federated_network.client import Client
from federated_network.utils import train_client_models


class FairFedDriftClientTests(unittest.TestCase):
    def make_client_for_fit(self):
        client = Client.__new__(Client)
        client.model = nn.Linear(1, 1, bias=False)
        client.trainloader = object()
        client.epochs = 2
        client.mini_batch_size = 4
        return client

    def test_local_fit_runs_before_during_and_after_drift(self):
        for is_drift, is_drift_end in ((False, False), (True, False), (False, True)):
            with self.subTest(is_drift=is_drift, is_drift_end=is_drift_end):
                client = self.make_client_for_fit()
                with patch('federated_network.client.train') as train:
                    client.fit(is_drift, is_drift_end, None, 4,
                               constants.RecoveryAlgorithm.FAIRFEDDRIFT, None)
                train.assert_called_once_with(client.model, client.trainloader, _epochs=client.epochs)

    def test_training_downloads_the_learned_server_model_before_fit(self):
        clients = []
        observed_weights = {}
        for client_id in (3, 8):
            client = SimpleNamespace(
                client_id=client_id,
                parent_server_id=None,
                model=nn.Linear(1, 1, bias=False),
                drift_recovery_method=constants.RecoveryAlgorithm.FAIRFEDDRIFT,
                fit=None,
                fit_fairfeddrift_history=Mock(),
                evaluate=Mock(return_value=(0.0, 1.0))
            )

            def record_fit(*args, _client=client):
                observed_weights[_client.client_id] = _client.model.weight.detach().clone()

            client.fit = Mock(side_effect=record_fit)
            clients.append(client)

        servers = []
        for position, weight in enumerate((2.0, 7.0)):
            model = nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                model.weight.fill_(weight)
            servers.append(SimpleNamespace(
                server_id=position,
                strategy=SimpleNamespace(strategy_name=constants.RecoveryAlgorithm.FAIRFEDDRIFT),
                model=model
            ))

        def assign_to_distinct_clusters(_servers, round_clients):
            round_clients[0].parent_server_id = 1
            round_clients[1].parent_server_id = 0

        drift = SimpleNamespace(is_drift=True, is_drift_end=False, drifted_client_indices=None, current_round=0)

        def prepare_runtime(_servers, round_clients, _round_idx):
            assign_to_distinct_clusters(_servers, round_clients)
            return {
                'server_models': {0: servers[0].model, 1: servers[1].model},
                'history_loaders': {3: {}, 8: {}},
                'history_sample_counts': {3: {}, 8: {}}
            }

        with patch('federated_network.utils.prepare_fairfeddrift_timestep', side_effect=prepare_runtime):
            train_client_models(
                clients, [3, 8], servers, drift,
                {'is_server_adaptability': False}, constants.RecoveryAlgorithm.FAIRFEDDRIFT)

        self.assertAlmostEqual(observed_weights[3].item(), 7.0)
        self.assertAlmostEqual(observed_weights[8].item(), 2.0)
        self.assertEqual([client.fairfeddrift_training_cluster_id for client in clients], [1, 0])
        self.assertEqual(clients[0].fit.call_args.args[2], servers[1].model.state_dict())
        self.assertEqual(clients[1].fit.call_args.args[2], servers[0].model.state_dict())
        clients[0].fit_fairfeddrift_history.assert_called_once_with({
            0: servers[0].model, 1: servers[1].model}, {}, {})
        clients[1].fit_fairfeddrift_history.assert_called_once_with({
            0: servers[0].model, 1: servers[1].model}, {}, {})

    def test_history_models_train_independently_without_mutating_client_or_servers(self):
        """Start one temporary model from each server and retain detached uploads."""
        client = Client.__new__(Client)
        client.model = nn.Linear(1, 1, bias=False)
        client.epochs = 3
        client.fairfeddrift_history_uploads = {}
        client.fairfeddrift_history_sample_counts = {}
        with torch.no_grad():
            client.model.weight.fill_(9.0)

        server_models = {4: nn.Linear(1, 1, bias=False), 8: nn.Linear(1, 1, bias=False)}
        with torch.no_grad():
            server_models[4].weight.fill_(1.0)
            server_models[8].weight.fill_(2.0)
        loaders = {
            4: DataLoader(TensorDataset(torch.tensor([[10.0]]), torch.tensor([0]))),
            8: DataLoader(TensorDataset(torch.tensor([[20.0]]), torch.tensor([0])))
        }
        trained_models = []
        starts = []

        def fake_train(model, loader, _epochs):
            trained_models.append(model)
            starts.append((model.weight.item(), loader.dataset.tensors[0][0].item(), _epochs))
            with torch.no_grad():
                model.weight.add_(loader.dataset.tensors[0][0])

        with patch('federated_network.client.train', side_effect=fake_train) as train:
            uploads = client.fit_fairfeddrift_history(server_models, loaders, {4: 1, 8: 1})

        self.assertEqual(starts, [(1.0, 10.0, 3), (2.0, 20.0, 3)])
        self.assertEqual(train.call_count, 2)
        self.assertIsNot(trained_models[0], trained_models[1])
        self.assertAlmostEqual(uploads[4]['weight'].item(), 11.0)
        self.assertAlmostEqual(uploads[8]['weight'].item(), 22.0)
        self.assertEqual(client.fairfeddrift_history_sample_counts, {4: 1, 8: 1})
        self.assertAlmostEqual(client.model.weight.item(), 9.0)
        self.assertAlmostEqual(server_models[4].weight.item(), 1.0)
        self.assertAlmostEqual(server_models[8].weight.item(), 2.0)

        with torch.no_grad():
            trained_models[0].weight.fill_(-5.0)
        self.assertAlmostEqual(uploads[4]['weight'].item(), 11.0)

    def test_history_training_rejects_inconsistent_counts_before_training(self):
        client = Client.__new__(Client)
        client.epochs = 1
        client.fairfeddrift_history_uploads = {99: 'stale'}
        client.fairfeddrift_history_sample_counts = {99: 1}
        server_models = {4: nn.Linear(1, 1)}
        loaders = {4: DataLoader(TensorDataset(torch.ones(2, 1), torch.zeros(2, dtype=torch.long)))}
        with patch('federated_network.client.train') as train:
            with self.assertRaisesRegex(ValueError, 'positive counts'):
                client.fit_fairfeddrift_history(server_models, loaders, {4: 1})
        train.assert_not_called()
        self.assertEqual(client.fairfeddrift_history_uploads, {})
        self.assertEqual(client.fairfeddrift_history_sample_counts, {})

    def test_empty_retained_history_clears_previous_round_uploads(self):
        client = Client.__new__(Client)
        client.epochs = 1
        client.fairfeddrift_history_uploads = {3: {'stale': torch.tensor(1.0)}}
        client.fairfeddrift_history_sample_counts = {3: 1}
        self.assertEqual(client.fit_fairfeddrift_history({}, {}, {}), {})
        self.assertEqual(client.fairfeddrift_history_sample_counts, {})


if __name__ == '__main__':
    unittest.main()
